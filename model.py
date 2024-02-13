import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import normal


class Actor(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim=256, num_layers=1):
        super(Actor, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.n_states = n_states
        self.n_actions = n_actions
        self.num_layers = num_layers

        self.fc1 = nn.Linear(in_features=self.n_states, out_features=hidden_dim)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)

        self.lstm = nn.LSTM(
            input_size=hidden_dim, hidden_size=hidden_dim, num_layers=self.num_layers
        )
        self.mu = nn.Linear(in_features=hidden_dim, out_features=n_actions)
        self.log_std = nn.Parameter(torch.zeros(1, self.n_actions))

        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                layer.bias.data.zero_()

    def forward(self, x, hidden_states):
        batch_size = x.size(0)

        hidden_states = initialize_hidden_states(
            hidden_states, batch_size, self.lstm, self.device
        )
        x = x.squeeze()
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = x.unsqueeze(0)
        output, hidden_states = self.lstm(x, hidden_states)
        mu = self.mu(output.squeeze(0))
        std = torch.exp(self.log_std + 1e-5)

        dist = normal.Normal(mu, std)

        return dist, hidden_states


class Critic(nn.Module):
    def __init__(self, n_states, hidden_dim=256, num_layers=1):
        super(Critic, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.n_states = n_states
        self.num_layers = num_layers

        self.fc1 = nn.Linear(in_features=self.n_states, out_features=hidden_dim)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim, hidden_size=hidden_dim, num_layers=self.num_layers
        )
        self.value_1 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.value_2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.value_3 = nn.Linear(in_features=hidden_dim, out_features=1)

        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                layer.bias.data.zero_()

    def forward(self, x, hidden_states):
        batch_size = x.size(0)

        hidden_states = initialize_hidden_states(
            hidden_states, batch_size, self.lstm, self.device
        )
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.unsqueeze(0)

        output, hidden_states = self.lstm(x, hidden_states)

        value = F.relu(self.value_1(output.squeeze(0)))
        if value.dim() == 1:
            value = value.unsqueeze(0)
        value = F.relu(self.value_2(value))
        value = self.value_3(value)

        return value, hidden_states


def initialize_hidden_states(hidden_states, batch_size, lstm, device):
    if hidden_states is None:
        # 각 hidden_state와 cell_state의 차원을 (batch_size, 1, lstm.hidden_size)로 설정
        hidden_state = torch.zeros(
            lstm.num_layers, batch_size, lstm.hidden_size, device=device
        )
        cell_state = torch.zeros(
            lstm.num_layers, batch_size, lstm.hidden_size, device=device
        )
        hidden_states = (hidden_state, cell_state)
    return hidden_states
