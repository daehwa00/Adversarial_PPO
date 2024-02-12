import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import normal


class Actor(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim=64):
        super(Actor, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.n_states = n_states
        self.n_actions = n_actions

        self.fc1 = nn.Linear(in_features=self.n_states, out_features=hidden_dim)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1)
        self.mu = nn.Linear(in_features=hidden_dim, out_features=n_actions)

        self.log_std = nn.Parameter(torch.zeros(1, self.n_actions))

        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                layer.bias.data.zero_()

    def forward(self, x, hidden_states):
        # 배치 차원이 있는 경우 (훈련 단계)
        batch_size = x.size(0)

        # hidden_states 처리
        hidden_states = self.initialize_hidden_states(
            hidden_states, batch_size, self.device
        )
        x = x.squeeze()
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = x.unsqueeze(0)
        output, hidden_state = self.lstm(x, hidden_states)

        # LSTM의 출력을 사용하여 mu 계산
        mu = self.mu(output.squeeze(0))

        # 표준 편차 계산
        std = torch.exp(self.log_std + 1e-5)

        # 정규 분포 생성
        dist = normal.Normal(mu, std)

        return dist, hidden_state

    def initialize_hidden_states(self, hidden_states, batch_size, device):
        if hidden_states is None:
            h_0 = torch.zeros(
                self.lstm.num_layers, batch_size, self.lstm.hidden_size
            ).to(device)
            c_0 = torch.zeros(
                self.lstm.num_layers, batch_size, self.lstm.hidden_size
            ).to(device)

            hidden_states = (h_0, c_0)
        return hidden_states


class Critic(nn.Module):
    def __init__(self, n_states, hidden_dim=256, seq_len=64):
        super(Critic, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.n_states = n_states
        self.seq_len = seq_len

        # One-hot encoding을 위해 seq_len을 사용하지 않고, 대신 torch.eye를 사용합니다.

        self.fc1 = nn.Linear(
            in_features=self.n_states + seq_len, out_features=hidden_dim
        )  # 입력 차원 수정: One-hot 벡터 크기 추가
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1)
        self.value_1 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.value_2 = nn.Linear(
            in_features=hidden_dim + seq_len, out_features=hidden_dim
        )
        self.value_3 = nn.Linear(in_features=hidden_dim + seq_len, out_features=1)

        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                layer.bias.data.zero_()

    def forward(self, x, time_step, hidden_states):
        batch_size = x.size(0)

        # One-hot encoding 생성
        one_hot = torch.zeros(batch_size, self.seq_len, device=self.device)
        one_hot[torch.arange(batch_size), time_step] = 1
        hidden_states = self.initialize_hidden_states(
            hidden_states, batch_size, self.device
        )

        # One-hot 벡터와 상태 정보를 결합
        x = torch.cat([x, one_hot], dim=-1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.unsqueeze(0)

        output, hidden_states = self.lstm(x, hidden_states)

        value = F.relu(self.value_1(output.squeeze(0)))
        value = torch.cat([value, one_hot], dim=-1)
        if value.dim() == 1:
            value = value.unsqueeze(0)
        value = F.relu(self.value_2(value))
        value = torch.cat([value, one_hot], dim=-1)
        value = self.value_3(value)

        return value, hidden_states

    def initialize_hidden_states(self, hidden_states, batch_size, device):
        if hidden_states is None:
            h_0 = torch.zeros(
                self.lstm.num_layers, batch_size, self.lstm.hidden_size
            ).to(device)
            c_0 = torch.zeros(
                self.lstm.num_layers, batch_size, self.lstm.hidden_size
            ).to(device)
            hidden_states = (h_0, c_0)
        return hidden_states
