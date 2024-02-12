import torch
import matplotlib.pyplot as plt


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
        returns,
        advs,
        values,
        log_probs,
        hidden_states_actor,
        hidden_states_critic,
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
                returns[indices],
                advs[indices],
                values[indices],
                log_probs[indices],
                hidden_states_actor[indices],
                hidden_states_critic[indices],
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
                return_,
                adv,
                old_value,
                old_log_prob,
                hidden_state_actor,
                hidden_state_critic,
                time_step,
            ) in self.choose_mini_batch(
                self.mini_batch_size,
                tensor_manager.states_tensor,
                tensor_manager.actions_tensor,
                returns,
                tensor_manager.advantages_tensor,
                tensor_manager.values_tensor,
                tensor_manager.log_probs_tensor,
                tensor_manager.hidden_states_actor_tensor,
                tensor_manager.hidden_states_critic_tensor,
                tensor_manager.time_step_tensor,
            ):
                h_a = hidden_state_actor[:, 0, :].unsqueeze(0).contiguous()
                c_a = hidden_state_actor[:, 1, :].unsqueeze(0).contiguous()
                h_c = hidden_state_critic[:, 0, :].unsqueeze(0).contiguous()
                c_c = hidden_state_critic[:, 1, :].unsqueeze(0).contiguous()
                hidden_state_actor = (h_a, c_a)
                hidden_state_critic = (h_c, c_c)

                # 업데이트된 숨겨진 상태를 사용하여 critic 및 actor 업데이트
                value, _ = self.agent.get_value(
                    state, hidden_state_critic, use_grad=True
                )

                value += -self.env.gamma * time_step.unsqueeze(1)

                critic_loss = (return_ - value).pow(2).mean()

                new_dist, _ = self.agent.choose_dists(
                    state, hidden_state_actor, use_grad=True
                )
                new_log_prob = new_dist.log_prob(action).sum(dim=1)
                ratio = (new_log_prob - old_log_prob).exp()

                actor_loss = self.compute_actor_loss(ratio, adv)

                self.agent.optimize(actor_loss, critic_loss)

        return actor_loss, critic_loss

    def step(self):

        for iteration in range(1, 1 + self.n_iterations):
            # Initialize the environment
            done_times = [-1] * self.env_num
            states = self.env.reset()
            tensor_manager = TensorManager(
                self.env_num,
                self.horizon,
                states.shape[1:],  # [channel, height, width]
                self.env.action_space,  # action space
                self.env.state_space,  # encoded state space
                self.agent.device,
            )
            hidden_states_actor = None
            hidden_states_critic = None
            self.time_step = 0

            # 1 episode (data collection)
            for t in range(self.horizon):
                # Actor
                dists, hidden_states_actor = self.agent.choose_dists(
                    states, hidden_states_actor, use_grad=False
                )
                actions = self.agent.choose_actions(dists)
                scaled_actions = self.agent.scale_actions(actions)
                log_prob = dists.log_prob(actions).sum(dim=1)

                # Critic
                value, hidden_states_critic = self.agent.get_value(
                    states, hidden_states_critic, use_grad=False
                )

                # apply action to the environment
                next_states, rewards, dones, _ = self.env.step(scaled_actions)

                tensor_manager.update_tensors(
                    states,
                    actions,
                    rewards,
                    value,
                    log_prob,
                    dones,
                    hidden_states_actor,
                    hidden_states_critic,
                    t,
                )

                for i in range(self.env_num):
                    if dones[i] and done_times[i] == -1:
                        done_times[i] = t + 1

                self.time_step += 1

            # 데이터 수집 단계 종료
            tensor_manager.filter_with_done_times(done_times)

            for i in range(self.env_num):
                # 환경이 끝나지 않았다면, 마지막 상태의 value를 계산
                if done_times[i] == -1:
                    tensor_manager.values_tensor[i, done_times[i]] = 0
                # 환경이 끝났다면, 마지막 상태의 value를 0으로 설정
                else:
                    h_c = tensor_manager.hidden_states_critic_tensor[
                        i, -1, 0, :
                    ].unsqueeze(0)
                    c_c = tensor_manager.hidden_states_critic_tensor[
                        i, -1, 1, :
                    ].unsqueeze(0)
                    next_value, _ = self.agent.get_value(
                        next_states[i].unsqueeze(0),
                        (h_c, c_c),
                        use_grad=False,
                    )
                    tensor_manager.values_tensor[i, -1] = next_value.squeeze()

            advs = self.get_gae(
                tensor_manager.rewards_tensor,
                tensor_manager.values_tensor,
                tensor_manager.dones_tensor,
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
    def get_gae(self, rewards, values, dones, done_times, gamma=1, lam=0.95):
        assert (
            rewards.ndim == 2 and values.ndim == 2 and dones.ndim == 2
        ), "Inputs must be 2D arrays."
        assert (
            values.shape[0] == rewards.shape[0]
            and values.shape[1] == rewards.shape[1] + 1
        ), "Values should have one more time step than rewards and dones."
        assert (
            len(done_times) == rewards.shape[0]
        ), "Length of done_times must match the number of environments."

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

        fig.subplots_adjust(hspace=0.3, wspace=0.3)  # 서브플롯 간격 조절
        plt.tight_layout()
        plt.savefig(f"results/results_graphs.png")
        plt.close()


class TensorManager:
    def __init__(self, env_num, horizon, states_shape, action_dim, hidden_dim, device):
        self.env_num = env_num
        self.horizon = horizon
        self.states_shape = states_shape
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.device = device

        self.states_tensor = self.init_tensor(
            [self.env_num, self.horizon, *self.states_shape], False
        )
        self.actions_tensor = self.init_tensor(
            [self.env_num, self.horizon, self.action_dim], False
        )
        self.rewards_tensor = self.init_tensor([self.env_num, self.horizon], False)
        self.values_tensor = self.init_tensor([self.env_num, self.horizon + 1], False)
        self.log_probs_tensor = self.init_tensor([self.env_num, self.horizon], False)
        self.dones_tensor = self.init_tensor([self.env_num, self.horizon + 1], False)
        self.hidden_states_actor_tensor = self.init_tensor(
            [self.env_num, self.horizon, 2, 64], False
        )
        self.hidden_states_critic_tensor = self.init_tensor(
            [self.env_num, self.horizon, 2, 64], False
        )
        self.advantages_tensor = self.init_tensor([self.env_num, self.horizon], False)
        self.time_step_tensor = torch.arange(
            0, self.horizon, device=self.device
        ).repeat(self.env_num, 1)

    def init_tensor(self, shape, requires_grad):
        return torch.zeros(shape, requires_grad=requires_grad).to(self.device)

    def reset_done_flags(self, done_times):
        for i, done_time in enumerate(done_times):
            if done_time != -1:
                self.dones_tensor[i, done_time:] = 1

    def update_tensors(
        self,
        states,
        actions,
        rewards,
        values,
        log_probs,
        dones,
        hidden_states_actor,
        hidden_states_critic,
        t,
    ):
        self.states_tensor[:, t] = states
        self.actions_tensor[:, t] = actions
        self.rewards_tensor[:, t] = rewards
        self.values_tensor[:, t] = values.squeeze()
        self.log_probs_tensor[:, t] = log_probs
        self.dones_tensor[:, t] = dones
        self.hidden_states_actor_tensor[:, t, 0, :] = hidden_states_actor[0].squeeze()
        self.hidden_states_actor_tensor[:, t, 1, :] = hidden_states_actor[1].squeeze()
        self.hidden_states_critic_tensor[:, t, 0, :] = hidden_states_critic[0].squeeze()
        self.hidden_states_critic_tensor[:, t, 1, :] = hidden_states_critic[1].squeeze()

    def reset_done_flags(self, done_times):
        for i, done_time in enumerate(done_times):
            if done_time != -1:
                self.dones_tensor[i, done_time:] = 1

    def filter_post_done_data(self, tensor, done_times):
        for i, done_time in enumerate(done_times):
            if done_time != -1:
                tensor[i, done_time:] = 0
        return tensor

    def filter_with_done_times(self, done_times):
        # done_times를 참고하여 dones_times 이후의 데이터를 필터링
        for i in range(self.env_num):
            if done_times[i] != -1:
                # 마지막 step 이후의 데이터를 0으로 설정
                self.actions_tensor[i, done_times[i] :] = 0
                self.rewards_tensor[i, done_times[i] :] = 0
                self.values_tensor[i, done_times[i] + 1 :] = 0
                self.log_probs_tensor[i, done_times[i] :] = 0
                self.dones_tensor[i, done_times[i] :] = 1
                self.hidden_states_actor_tensor[i, done_times[i] + 1 :, :, :] = 0
                self.hidden_states_critic_tensor[i, done_times[i] + 1 :, :, :] = 0

    def filter_post_done_data(self, done_times):
        # 각 환경의 done_time을 기반으로 필터링된 텐서의 총 길이 계산
        remaining_steps = sum(self.horizon if t == -1 else t for t in done_times)

        # 모든 관련 텐서에 대한 필터링 수행
        self.states_tensor = self._filter_tensor(
            self.states_tensor, done_times, remaining_steps
        )
        self.actions_tensor = self._filter_tensor(
            self.actions_tensor, done_times, remaining_steps
        )
        self.rewards_tensor = self._filter_tensor(
            self.rewards_tensor, done_times, remaining_steps, is_flat=True
        )
        self.values_tensor = self._filter_tensor(
            self.values_tensor, done_times, remaining_steps, is_flat=True
        )
        self.log_probs_tensor = self._filter_tensor(
            self.log_probs_tensor, done_times, remaining_steps, is_flat=True
        )
        self.dones_tensor = self._filter_tensor(
            self.dones_tensor, done_times, remaining_steps, is_flat=True
        )
        self.hidden_states_actor_tensor = self._filter_tensor(
            self.hidden_states_actor_tensor, done_times, remaining_steps
        )
        self.hidden_states_critic_tensor = self._filter_tensor(
            self.hidden_states_critic_tensor, done_times, remaining_steps
        )
        self.advantages_tensor = self._filter_tensor(
            self.advantages_tensor, done_times, remaining_steps, is_flat=True
        )
        self.time_step_tensor = self._filter_tensor(
            self.time_step_tensor, done_times, remaining_steps, is_flat=True
        )

    def _filter_tensor(self, tensor, done_times, remaining_steps, is_flat=False):
        # 필터링된 텐서의 초기화
        if is_flat:
            filtered_tensor = torch.zeros([remaining_steps], device=self.device)
        else:
            filtered_tensor = torch.zeros(
                [remaining_steps, *tensor.shape[2:]], device=self.device
            )

        start_idx = 0
        for i, done_time in enumerate(done_times):
            length = self.horizon if done_time == -1 else done_time
            end_idx = start_idx + length
            if is_flat:
                filtered_tensor[start_idx:end_idx] = tensor[i, :length].clone()
            else:
                filtered_tensor[start_idx:end_idx, ...] = tensor[
                    i, :length, ...
                ].clone()
            start_idx = end_idx

        return filtered_tensor
