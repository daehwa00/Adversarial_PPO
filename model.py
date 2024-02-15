import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import normal


class Actor(nn.Module):
    def __init__(self, n_states, n_actions, action_map_size, hidden_dim=256):
        super(Actor, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.n_states = n_states
        self.n_actions = n_actions
        self.action_map_size = action_map_size

        self.state_fc = nn.Linear(in_features=self.n_states, out_features=hidden_dim)

        self.action_map_fc = nn.Linear(
            in_features=self.action_map_size, out_features=hidden_dim
        )

        self.combined_fc = nn.Linear(
            in_features=hidden_dim * 2, out_features=hidden_dim
        )

        self.mu = nn.Linear(in_features=hidden_dim, out_features=n_actions)
        self.log_std = nn.Parameter(torch.zeros(1, self.n_actions))

        self._init_weights()

    def _init_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                layer.bias.data.zero_()

    def forward(self, x, action_map):
        batch_size = x.size(0)

        x = x.squeeze()
        state_emb = F.relu(self.state_fc(x))

        action_map_emb = torch.tanh(
            self.action_map_fc(action_map.view(-1, self.action_map_size))
        )

        combined_emb = torch.cat([state_emb, action_map_emb], dim=1)
        combined_emb = F.relu(self.combined_fc(combined_emb))
        x = x.unsqueeze(0)
        mu = self.mu(combined_emb.squeeze(0))
        std = torch.exp(self.log_std + 1e-5)

        dist = normal.Normal(mu, std)

        return dist


class Critic(nn.Module):
    def __init__(self, n_states, action_map_size, hidden_dim=256):
        super(Critic, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.n_states = n_states
        self.action_map_size = action_map_size

        self.state_fc = nn.Linear(in_features=self.n_states, out_features=hidden_dim)
        self.action_map_fc = nn.Linear(
            in_features=action_map_size, out_features=hidden_dim
        )
        self.combined_fc1 = nn.Linear(
            in_features=hidden_dim * 2, out_features=hidden_dim
        )
        self.combined_fc2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)

        # 최종 가치 출력을 위한 레이어
        self.value_head = nn.Linear(in_features=hidden_dim, out_features=1)

        self.cls_token = nn.Parameter(torch.randn(1, hidden_dim))
        self.postions = nn.Parameter(torch.randn(32 * 32 + 1, hidden_dim))

        self._init_weights()

    def _init_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                layer.bias.data.zero_()

    def forward(self, x, action_map):
        batch_size = x.size(0)

        state_emb = F.relu(self.state_fc(x))

        action_map_emb = F.relu(
            self.action_map_fc(action_map.view(-1, self.action_map_size))
        )

        combined_emb = torch.cat([state_emb, action_map_emb], dim=1)
        if combined_emb.dim() == 1:
            combined_emb = combined_emb.unsqueeze(0)
        combined_emb = F.relu(self.combined_fc1(combined_emb))
        combined_emb = F.relu(self.combined_fc2(combined_emb))
        value = self.value_head(combined_emb)
        return value
