from model import Actor, Critic
from torch.optim import Adam
from torch import from_numpy
import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR


class Agent:
    def __init__(self, env_name, n_iter, n_states, n_actions, lr):
        self.env_name = env_name
        self.n_iter = n_iter
        self.n_states = n_states
        self.n_actions = n_actions
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.lr = lr

        self.actor = Actor(n_states=self.n_states, n_actions=self.n_actions).to(
            self.device
        )
        self.critic = Critic(n_states=self.n_states + 16).to(self.device)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.lr, eps=1e-5)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.lr, eps=1e-5)

        self.critic_loss = torch.nn.MSELoss()

        self.scheduler = lambda step: max(1.0 - float(step / self.n_iter), 1e-5)

        self.actor_scheduler = LambdaLR(self.actor_optimizer, lr_lambda=self.scheduler)
        self.critic_scheduler = LambdaLR(self.actor_optimizer, lr_lambda=self.scheduler)

    def choose_dist(self, state, hidden_state=None):
        state = np.expand_dims(state, 0)
        state = from_numpy(state).float().to(self.device)
        with torch.no_grad():
            dist, hidden_state = self.actor(state, hidden_state)

        return dist, hidden_state

    def get_value(self, state, hidden_state=None):
        state = np.expand_dims(state, 0)
        state = from_numpy(state).float().to(self.device)
        with torch.no_grad():
            value, hidden_state = self.critic(state, hidden_state)
        return value.detach().cpu().numpy(), hidden_state

    def optimize(self, actor_loss, critic_loss):
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def schedule_lr(self, actor_loss, critic_loss):
        self.actor_scheduler.step()
        self.critic_scheduler.step()

    def save_weights(self, iteration, state_rms):
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
                "actor_scheduler_state_dict": self.actor_scheduler.state_dict(),
                "critic_scheduler_state_dict": self.critic_scheduler.state_dict(),
                "iteration": iteration,
            },
            self.env_name + "_weights.pth",
        )

    def load_weights(self):
        checkpoint = torch.load(self.env_name + "_weights.pth")
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        self.actor_scheduler.load_state_dict(checkpoint["actor_scheduler_state_dict"])
        self.critic_scheduler.load_state_dict(checkpoint["critic_scheduler_state_dict"])
        iteration = checkpoint["iteration"]
        state_rms_mean = checkpoint["state_rms_mean"]
        state_rms_var = checkpoint["state_rms_var"]

        return iteration, state_rms_mean, state_rms_var

    def set_to_eval_mode(self):
        self.actor.eval()
        self.critic.eval()

    def set_to_train_mode(self):
        self.actor.train()
        self.critic.train()


class EncodedAgent(Agent):
    def __init__(
        self,
        encoder,
        env_name,
        n_iter,
        n_states,
        n_actions,
        lr,
    ):
        super(EncodedAgent, self).__init__(env_name, n_iter, n_states, n_actions, lr)
        self.encoder = encoder
        self.encoder.eval()  # Encoder를 평가 모드로 설정
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_encoded_state(self, state):
        with torch.no_grad():
            encoded_state = self.encoder(state)

        return encoded_state.squeeze(0)

    def choose_dists(self, state, hidden_state=None):
        encoded_state = self.get_encoded_state(state).squeeze()
        with torch.no_grad():
            dist, hidden_state = self.actor(encoded_state, hidden_state)
        return dist, hidden_state

    def get_value(
        self,
        state,
        position,
        hidden_state=None,
    ):
        encoded_state = self.get_encoded_state(state).squeeze()
        # concat
        if encoded_state.dim() == 1:
            encoded_state = encoded_state.unsqueeze(0)
            position = position.unsqueeze(0)
        encoded_state = torch.cat((encoded_state, position), dim=1)

        with torch.no_grad():
            value, hidden_state = self.critic(encoded_state, hidden_state)
        return value, hidden_state

    def choose_actions(self, dist):
        action = dist.sample()
        return action

    def scale_actions(self, actions):
        scaled_actions = torch.zeros_like(actions)
        # RGB 값을 0-1로 스케일링
        scaled_actions[:, :3] = (torch.clamp(actions[:, :3], -1, 1) * 255).int() / 255.0

        # XY 값을 -1-1로 스케일링
        scaled_actions[:, 3:] = (
            (torch.clamp(actions[:, 3:], -1, 1) + 1.0) / 2.0 * 31
        ).int()

        return scaled_actions

    def critic_forward(self, state, position, hidden_state=None):
        encoded_state = self.get_encoded_state(state).squeeze()
        position = position.squeeze()
        # concat
        if encoded_state.dim() == 1:
            encoded_state = encoded_state.unsqueeze(0)
            position = position.unsqueeze(0)
        encoded_state = torch.cat((encoded_state, position), dim=1)
        value, hidden_state = self.critic.forward(encoded_state, hidden_state)

        return value, hidden_state

    def actor_forward(self, state, hidden_state=None):
        encoded_state = self.get_encoded_state(state)
        dist, hidden_state = self.actor.forward(encoded_state, hidden_state)

        return dist, hidden_state
