from model import Actor, Critic

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
import math


class Agent:
    def __init__(
        self,
        env_name,
        n_iter,
        n_actions,
        action_map_size,
        hidden_dim,
        n_layers,
        num_heads,
        lr,
        warmup_steps,
    ):
        self.env_name = env_name
        self.n_iter = n_iter
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.action_map_size = action_map_size
        self.channel, self.height, self.width = action_map_size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.lr = lr
        self.num_heads = num_heads
        self.n_layers = n_layers
        self.warmup_steps = warmup_steps

        # Actor
        self.actor = Actor(
            n_actions=self.n_actions,
            image_size=self.height,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            num_heads=self.num_heads,
        ).to(self.device)

        # Critic
        self.critic = Critic(
            image_size=self.height,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            num_heads=self.num_heads,
        ).to(self.device)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr, eps=1e-8)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr * 5, eps=1e-8)

        self.critic_loss = torch.nn.MSELoss()

        self.actor_scheduler = self.configure_scheduler(self.actor_optimizer)
        self.critic_scheduler = self.configure_scheduler(self.critic_optimizer)

    def configure_scheduler(self, optimizer):
        def lr_lambda(current_step: int):
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            return max(
                0.0,
                0.5
                * (
                    1.0
                    + torch.cos(
                        math.pi
                        * (current_step - self.warmup_steps)
                        / (self.n_iter - self.warmup_steps)
                    )
                ),
            )

        scheduler = LambdaLR(optimizer, lr_lambda)
        return scheduler

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

    def choose_dists(self, state, action_map, use_grad=True):
        if use_grad:
            dist = self.actor(state, action_map)
        else:
            with torch.no_grad():
                dist = self.actor(state, action_map)
        return dist

    def get_value(self, state, action_map, use_grad=True):
        # concat
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if use_grad:
            value = self.critic(state, action_map)

        else:
            with torch.no_grad():
                value = self.critic(state, action_map)
        return value

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
