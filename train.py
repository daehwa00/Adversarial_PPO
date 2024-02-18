import torch
import matplotlib.pyplot as plt
from tensormanager import TensorManager


class Train:
    def __init__(
        self,
        env,
        env_name,
        n_iterations,
        agent,
        epochs,
        mini_batch_size,
        epsilon,
        horizon,
    ):
        self.env = env
        self.env_name = env_name
        self.agent = agent
        self.epsilon = epsilon
        self.horizon = horizon
        self.epochs = epochs
        self.mini_batch_size = mini_batch_size
        self.n_iterations = n_iterations
        self.env_num = env.env_batch
        self.start_time = 0
        self.running_reward = 0
        self.steps_history = []
        self.rewards_history = []
        self.actor_loss_history = []
        self.critic_loss_history = []

    def choose_mini_batch(
        self,
        mini_batch_size,
        states,
        actions,
        action_maps,
        returns,
        advs,
        values,
        log_probs,
        time_step_tensor,
    ):
        full_batch_size = states.size(0)
        for _ in range(full_batch_size // mini_batch_size):
            # 무작위로 mini_batch_size 개의 인덱스를 선택
            indices = torch.randperm(full_batch_size)[:mini_batch_size].to(
                states.device
            )
            yield (
                states[indices],
                actions[indices],
                action_maps[indices],
                returns[indices],
                advs[indices],
                values[indices],
                log_probs[indices],
                time_step_tensor[indices],
            )

    def train(
        self,
        tensor_manager,
    ):
        returns = tensor_manager.advantages_tensor + tensor_manager.values_tensor

        for epoch in range(self.epochs):
            for (
                state,
                action,
                action_map,
                return_,
                adv,
                old_value,
                old_log_prob,
                time_step,
            ) in self.choose_mini_batch(
                self.mini_batch_size,
                tensor_manager.states_tensor,
                tensor_manager.actions_tensor,
                tensor_manager.action_maps_tensor,
                returns,
                tensor_manager.advantages_tensor,
                tensor_manager.values_tensor,
                tensor_manager.log_probs_tensor,
                tensor_manager.time_step_tensor,
            ):

                # 업데이트된 숨겨진 상태를 사용하여 critic 및 actor 업데이트
                value = self.agent.get_value(state, action_map, use_grad=True)

                critic_loss = (return_ - value).pow(2).mean()

                new_dist = self.agent.choose_dists(state, action_map, use_grad=True)
                new_log_prob = new_dist.log_prob(action).sum(dim=1)
                ratio = (new_log_prob - old_log_prob).exp()

                actor_loss = self.compute_actor_loss(ratio, adv)

                entropy_loss = new_dist.entropy().mean()

                actor_loss += -0.01 * entropy_loss

                self.agent.optimize(actor_loss, critic_loss)

        return actor_loss, critic_loss

    def step(self):
        for iteration in range(1, 1 + self.n_iterations):
            # Initialize the environment
            done_times = [-1] * self.env_num
            states = self.env.reset()
            tensor_manager = TensorManager(
                env_num=self.env_num,
                horizon=self.horizon,
                image_shape=states.shape[1:],  # [channel, height, width]
                action_dim=self.env.action_space,  # action space
                device=self.agent.device,
            )

            prev_action_map = torch.zeros(
                self.env_num, *states.shape[1:], device=self.agent.device
            )
            # 1 episode (data collection)
            for t in range(self.horizon):
                # Actor
                dists = self.agent.choose_dists(states, prev_action_map, use_grad=False)
                actions = self.agent.choose_actions(dists)
                scaled_actions = self.agent.scale_actions(actions)
                log_prob = dists.log_prob(actions).sum(dim=1)

                # Critic
                value = self.agent.get_value(states, prev_action_map, use_grad=False)

                # apply action to the environment
                next_states, action_map, rewards, dones = self.env.step(scaled_actions)

                tensor_manager.update_tensors(
                    states,
                    prev_action_map,
                    actions,
                    rewards,
                    value,
                    log_prob,
                    dones,
                    t,
                )

                for i in range(self.env_num):
                    if dones[i] and done_times[i] == -1:
                        done_times[i] = t + 1

                prev_action_map = action_map

            # 데이터 수집 단계 종료
            tensor_manager.filter_with_done_times(done_times)

            for i in range(self.env_num):
                # if the epsisode is not done
                if done_times[i] == -1:
                    next_value = self.agent.get_value(
                        tensor_manager.states_tensor[i, -1].unsqueeze(0),
                        tensor_manager.action_maps_tensor[i, -1].unsqueeze(0),
                        use_grad=False,
                    )
                    tensor_manager.values_tensor[i, done_times[i]] = next_value

                else:
                    tensor_manager.values_tensor[i, done_times[i]] = 0

            advs = self.get_gae(
                tensor_manager,
                done_times,
            )
            tensor_manager.advantages_tensor = advs
            tensor_manager.filter_post_done_data(done_times)

            # Train the agent
            actor_loss, critic_loss = self.train(tensor_manager)
            eval_rewards = torch.sum(
                tensor_manager.rewards_tensor
            )  # 각 환경별 총 보상 계산

            remaining_steps = sum(self.horizon if t == -1 else t for t in done_times)
            self.agent.schedule_lr(actor_loss, critic_loss)
            self.print_logs(
                iteration, actor_loss, critic_loss, eval_rewards, remaining_steps
            )

    # lambda가 작으면 TD(λ)의 편향이 커지고, 크면 MC에 가까워짐
    # GAE의 장점은 λ를 통해 편향-분산 트레이드오프를 조절할 수 있다는 것
    def get_gae(self, tensor_manager, done_times, gamma=1, lam=0.95):
        rewards = tensor_manager.rewards_tensor
        values = tensor_manager.values_tensor
        dones = tensor_manager.dones_tensor
        num_envs, horizon = rewards.shape
        advs = torch.zeros_like(rewards).to(rewards.device)

        # Adjusting the values after the end of the episodes
        for env_idx in range(num_envs):
            gae = 0
            for t in reversed(range(done_times[env_idx])):
                delta = (
                    rewards[env_idx, t]
                    + gamma * values[env_idx, t + 1] * (1 - dones[env_idx, t])
                    - values[env_idx, t]
                )
                gae = delta + gamma * lam * (1 - dones[env_idx, t]) * gae
                advs[env_idx, t] = gae
        return advs

    def compute_actor_loss(self, ratio, adv):
        pg_loss1 = adv * ratio
        pg_loss2 = adv * torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
        loss = -torch.min(pg_loss1, pg_loss2).mean()

        return loss

    def print_logs(self, iteration, actor_loss, critic_loss, eval_rewards, steps):

        if iteration == 1:
            self.running_reward = eval_rewards
        else:
            self.running_reward = self.running_reward * 0.99 + eval_rewards * 0.01
        running_reward = torch.mean(self.running_reward)
        current_actor_lr = self.agent.actor_optimizer.param_groups[0]["lr"]
        current_critic_lr = self.agent.critic_optimizer.param_groups[0]["lr"]

        self.steps_history.append(steps / self.env_num)
        self.rewards_history.append(running_reward.item())
        self.actor_loss_history.append(actor_loss.item())
        self.critic_loss_history.append(critic_loss.item())

        actor_loss = actor_loss.item() if torch.is_tensor(actor_loss) else actor_loss
        critic_loss = (
            critic_loss.item() if torch.is_tensor(critic_loss) else critic_loss
        )
        # eval_rewards의 평균을 계산
        if torch.is_tensor(eval_rewards):
            eval_rewards_val = eval_rewards.mean().item()
        else:
            eval_rewards_val = eval_rewards

        running_reward_val = torch.mean(self.running_reward).item()
        self.plot_and_save()

        if iteration % 5 == 0:
            print(
                f"Iter:{iteration}| "
                f"Ep_Reward:{eval_rewards_val:.3f}| "
                f"Running_reward:{running_reward_val:.3f}| "
                f"Actor_Loss:{actor_loss:.3f}| "
                f"Critic_Loss:{critic_loss:.3f}| "
                f"Actor_lr:{current_actor_lr}| "
                f"Critic_lr:{current_critic_lr}| "
                f"Steps:{steps/self.env_num:.3f}"
            )

    def plot_and_save(self):
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))  # 서브플롯의 크기를 늘림
        axs[0, 0].plot(self.steps_history, label="Average Steps")
        axs[0, 0].set_title("Average Steps")
        axs[0, 1].plot(self.rewards_history, label="Running Reward")
        axs[0, 1].set_title("Running Reward")
        axs[1, 0].plot(self.actor_loss_history, label="Actor Loss")
        axs[1, 0].set_title("Actor Loss")
        axs[1, 1].plot(self.critic_loss_history, label="Critic Loss")
        axs[1, 1].set_title("Critic Loss")

        for ax in axs.flat:
            ax.set(xlabel="Iteration", ylabel="Value")
            ax.label_outer()
            if ax.has_data():
                ax.legend()

        fig.subplots_adjust(hspace=1, wspace=1)  # 서브플롯 간격 조절
        plt.tight_layout()
        plt.savefig(f"results/results_graphs.png")
        plt.close()
