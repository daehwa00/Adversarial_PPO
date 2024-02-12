from torch import nn
from torch.distributions import normal
import torch


class Actor(nn.Module):
    def __init__(self, n_states, n_actions, hidden_size=64):
        super(Actor, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.n_states = n_states
        self.n_actions = n_actions

        self.fc1 = nn.Linear(in_features=self.n_states, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=64)
        self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_size, num_layers=1)
        self.mu = nn.Linear(in_features=hidden_size, out_features=n_actions)

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
    def __init__(self, n_states, hidden_size=64):
        super(Critic, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.n_states = n_states

        self.fc1 = nn.Linear(in_features=self.n_states, out_features=hidden_size)
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size, hidden_size=hidden_size, num_layers=1
        )
        self.value_1 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.value_2 = nn.Linear(in_features=hidden_size, out_features=1)

        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                layer.bias.data.zero_()

    def forward(self, x, hidden_states):
        batch_size = x.size(0)
        # hidden_states 처리

        hidden_states = self.initialize_hidden_states(
            hidden_states, batch_size, self.device
        )
        x = x.squeeze()
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = x.unsqueeze(0)

        output, hidden_states = self.lstm(x, hidden_states)

        value = torch.relu(self.value_1(output.squeeze(0)))
        if value.dim() == 1:
            value = value.unsqueeze(0)
        value = self.value_2(value)

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
