import torch
import numpy as np
import time
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
            )

    def train(
        self,
        states,
        actions,
        advs,
        values,
        log_probs,
        hidden_states_actor,
        hidden_states_critic,
    ):
        returns = advs + values

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
            ) in self.choose_mini_batch(
                self.mini_batch_size,
                states,
                actions,
                returns,
                advs,
                values,
                log_probs,
                hidden_states_actor,
                hidden_states_critic,
            ):
                h_a = hidden_state_actor[:, 0, :].unsqueeze(0).contiguous()
                c_a = hidden_state_actor[:, 1, :].unsqueeze(0).contiguous()
                h_c = hidden_state_critic[:, 0, :].unsqueeze(0).contiguous()
                c_c = hidden_state_critic[:, 1, :].unsqueeze(0).contiguous()
                hidden_state_actor = (h_a, c_a)
                hidden_state_critic = (h_c, c_c)

                # 업데이트된 숨겨진 상태를 사용하여 critic 및 actor 업데이트
                value, _ = self.agent.critic_forward(state, hidden_state_critic)

                critic_loss = (value - adv).pow(2).mean()

                new_dist, _ = self.agent.actor_forward(state, hidden_state_actor)
                new_log_prob = new_dist.log_prob(action).sum(dim=1)
                ratio = (new_log_prob - old_log_prob).exp()

                actor_loss = self.compute_actor_loss(ratio, adv)
                self.agent.optimize(actor_loss, critic_loss)

        return actor_loss, critic_loss

    def step(self):
        # iteration 횟수만큼 반복
        for iteration in range(1, 1 + self.n_iterations):
            done_times = [-1] * self.env_num
            states = self.env.reset()  # 환경 초기화
            states_tensor = torch.zeros(
                [self.env_num, self.horizon, *states.shape[1:]], requires_grad=False
            ).to(self.agent.device)
            actions_tensor = torch.zeros(
                [self.env_num, self.horizon, 5], requires_grad=False
            ).to(self.agent.device)
            rewards_tensor = torch.zeros(
                [self.env_num, self.horizon], requires_grad=False
            ).to(self.agent.device)
            values_tensor = torch.zeros(
                [self.env_num, self.horizon + 1], requires_grad=False
            ).to(self.agent.device)
            log_probs_tensor = torch.zeros(
                [self.env_num, self.horizon], requires_grad=False
            ).to(self.agent.device)
            dones_tensor = torch.zeros(
                [self.env_num, self.horizon + 1], requires_grad=False
            ).to(self.agent.device)
            hidden_states_actor_tensor = torch.zeros(
                [self.env_num, self.horizon, 2, 64], requires_grad=False
            ).to(self.agent.device)
            hidden_states_critic_tensor = torch.zeros(
                [self.env_num, self.horizon, 2, 64], requires_grad=False
            ).to(self.agent.device)

            hidden_states_actor = None
            hidden_states_critic = None

            self.start_time = time.time()

            # 1 episode (데이터 수집 단계)
            for t in range(self.horizon):
                # Actor
                dist, hidden_states_actor = self.agent.choose_dists(
                    states, hidden_states_actor
                )
                actions = self.agent.choose_actions(dist)
                scaled_actions = self.agent.scale_actions(actions)
                log_prob = dist.log_prob(actions).sum(dim=1)

                # Critic
                value, hidden_states_critic = self.agent.get_value(
                    states, hidden_states_critic
                )
                next_states, rewards, dones, _ = self.env.step(scaled_actions)

                # img = (next_states[0] + 1) / 2
                # plt.imshow(img.cpu().numpy().transpose(1, 2, 0))
                # plt.show()
                # plt.pause(1)

                h_a, c_a = hidden_states_actor
                h_c, c_c = hidden_states_critic

                states_tensor[:, t] = states
                actions_tensor[:, t] = actions
                rewards_tensor[:, t] = rewards
                values_tensor[:, t] = value.squeeze()
                log_probs_tensor[:, t] = log_prob
                dones_tensor[:, t] = dones
                hidden_states_actor_tensor[:, t, 0, :] = h_a.squeeze()
                hidden_states_actor_tensor[:, t, 1, :] = c_a.squeeze()
                hidden_states_critic_tensor[:, t, 0, :] = h_c.squeeze()
                hidden_states_critic_tensor[:, t, 1, :] = c_c.squeeze()

                for i in range(self.env_num):
                    if dones[i] and done_times[i] == -1:
                        done_times[i] = t + 1

            # done_times를 참고하여 dones_times 이후의 데이터를 필터링
            for i in range(self.env_num):
                if done_times[i] != -1:
                    # 마지막 step 이후의 데이터를 0으로 설정
                    actions_tensor[i, done_times[i] :] = 0
                    rewards_tensor[i, done_times[i] :] = 0
                    values_tensor[i, done_times[i] + 1 :] = 0
                    log_probs_tensor[i, done_times[i] :] = 0
                    dones_tensor[i, done_times[i] :] = 1
                    hidden_states_actor_tensor[i, done_times[i] + 1 :, :, :] = 0
                    hidden_states_critic_tensor[i, done_times[i] + 1 :, :, :] = 0

            remaining_steps = np.sum(
                [self.horizon if t == -1 else t for t in done_times]
            )

            for i in range(self.env_num):
                # 환경이 끝났다면, 마지막 상태의 value를 0으로 설정
                if done_times[i] != -1:
                    values_tensor[i, done_times[i]] = 0
                # 환경이 끝나지 않았다면, 마지막 상태의 value를 계산
                else:
                    h_c = hidden_states_critic_tensor[i, -1, 0, :].unsqueeze(0)
                    c_c = hidden_states_critic_tensor[i, -1, 1, :].unsqueeze(0)
                    next_value, _ = self.agent.get_value(
                        next_states[i].unsqueeze(0),
                        (h_c, c_c),
                    )
                    values_tensor[i, -1] = next_value.squeeze()

            advs = self.get_gae(rewards_tensor, values_tensor, dones_tensor, done_times)

            states_tensor = self.filter_post_done_data(
                states_tensor, done_times, self.env_num, remaining_steps
            )
            actions_tensor = self.filter_post_done_data(
                actions_tensor, done_times, self.env_num, remaining_steps
            )
            advs = self.filter_post_done_data(
                advs, done_times, self.env_num, remaining_steps
            )
            values_tensor = self.filter_post_done_data(
                values_tensor, done_times, self.env_num, remaining_steps
            )
            log_probs_tensor = self.filter_post_done_data(
                log_probs_tensor,
                done_times,
                self.env_num,
                remaining_steps,
            )
            hidden_states_actor_tensor = self.filter_post_done_data(
                hidden_states_actor_tensor,
                done_times,
                self.env_num,
                remaining_steps,
            )
            hidden_states_critic_tensor = self.filter_post_done_data(
                hidden_states_critic_tensor,
                done_times,
                self.env_num,
                remaining_steps,
            )

            actor_loss, critic_loss = self.train(
                states_tensor,
                actions_tensor,
                advs,
                values_tensor,
                log_probs_tensor,
                hidden_states_actor_tensor,
                hidden_states_critic_tensor,
            )
            eval_rewards = torch.sum(rewards_tensor, dim=1)  # 각 환경별 총 보상 계산

            self.agent.schedule_lr(actor_loss, critic_loss)
            self.print_logs(
                iteration, actor_loss, critic_loss, eval_rewards, remaining_steps
            )

    # lambda가 작으면 TD(λ)의 편향이 커지고, 크면 MC에 가까워짐
    def get_gae(self, rewards, values, dones, done_times, gamma=0.99, lam=0.95):
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
                # print(f"done:", dones[env_idx, t])
                # print(f"reward:", rewards[env_idx, t])
                # print(f"value:", values[env_idx, t])
                # print(f"next_value:", values[env_idx, t + 1])
                # print(f"t:", t)
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
        loss = torch.min(pg_loss1, pg_loss2).mean()
        loss = -loss

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

    def filter_post_done_data(self, tensor, done_times, env_num, remaining_steps):
        # 필터링된 텐서의 초기화
        if tensor.ndim == 2:
            filtered_tensor = torch.zeros([remaining_steps], device=tensor.device)
        else:
            filtered_tensor = torch.zeros(
                [remaining_steps, *tensor.shape[2:]], device=tensor.device
            )

        start_idx, end_idx = 0, 0

        for i in range(env_num):
            # 각 환경에서 끝나는 시점 계산
            length = self.horizon if done_times[i] == -1 else done_times[i]
            end_idx += length

            # 필요한 데이터를 추출하고 복사
            filtered_tensor[start_idx : start_idx + length] = tensor[i, :length]

            start_idx = end_idx
        return filtered_tensor
