import torch


class TensorManager:
    def __init__(
        self,
        env_num,
        horizon,
        image_shape,
        action_dim,
        device,
    ):
        self.env_num = env_num
        self.horizon = horizon
        self.image_shape = image_shape
        self.action_dim = action_dim
        self.device = device

        self.states_tensor = self.init_tensor(
            [self.env_num, self.horizon, *self.image_shape], False
        )
        self.action_maps_tensor = self.init_tensor(
            [self.env_num, self.horizon, *self.image_shape], False
        )
        self.actions_tensor = self.init_tensor(
            [self.env_num, self.horizon, self.action_dim], False
        )
        self.rewards_tensor = self.init_tensor([self.env_num, self.horizon], False)
        self.values_tensor = self.init_tensor([self.env_num, self.horizon + 1], False)
        self.log_probs_tensor = self.init_tensor([self.env_num, self.horizon], False)
        self.dones_tensor = self.init_tensor([self.env_num, self.horizon + 1], False)
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
        action_maps,
        actions,
        rewards,
        values,
        log_probs,
        dones,
        t,
    ):
        self.states_tensor[:, t] = states
        self.action_maps_tensor[:, t] = action_maps
        self.actions_tensor[:, t] = actions
        self.rewards_tensor[:, t] = rewards
        self.values_tensor[:, t] = values.squeeze()
        self.log_probs_tensor[:, t] = log_probs
        self.dones_tensor[:, t] = dones

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
                self.states_tensor[i, done_times[i] + 1 :] = 0
                self.actions_tensor[i, done_times[i] :] = 0
                self.action_maps_tensor[i, done_times[i] :] = 0
                self.rewards_tensor[i, done_times[i] :] = 0
                self.values_tensor[i, done_times[i] + 1 :] = 0
                self.log_probs_tensor[i, done_times[i] :] = 0
                self.dones_tensor[i, done_times[i] :] = 1

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
        self.action_maps_tensor = self._filter_tensor(
            self.action_maps_tensor, done_times, remaining_steps
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
