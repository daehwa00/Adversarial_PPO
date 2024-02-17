from model import Actor, Critic
from torch.optim import Adam
from torch import from_numpy
import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR


class Agent:
    def __init__(
        self,
        env_name,
        n_iter,
        n_actions,
        action_map_size,
        hidden_dim,
        n_layers,
        base_lr,
        warmup_steps,
    ):
        self.env_name = env_name
        self.n_iter = n_iter
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.action_map_size = action_map_size
        self.channel, self.height, self.width = action_map_size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.cnn_lr = base_lr * 10
        self.other_lr = base_lr
        self.warmup_steps = warmup_steps

        # Actor
        self.actor = Actor(
            n_actions=self.n_actions,
            image_size=self.height,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
        ).to(self.device)

        # Critic
        self.critic = Critic(
            image_size=self.height,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
        ).to(self.device)

        actor_params = [
            {
                "params": [
                    *list(self.actor.action_map_fc.parameters()),
                    *list(self.actor.layer1.parameters()),
                    *list(self.actor.layer2.parameters()),
                    *list(self.actor.layer3.parameters()),
                    *list(self.actor.mu.parameters()),
                    self.actor.log_std,
                ],
                "lr": self.cnn_lr,
            },
            {
                "params": [
                    self.actor.cls_token,
                    self.actor.position_embedding,
                    *list(
                        self.actor.cross_attention_layers.parameters()
                    ),  # 리스트 변환
                    *list(self.actor.layer_norms.parameters()),  # 리스트 변환
                ],
                "lr": self.other_lr,
            },  # cls_token과 position_embedding 학습률
        ]

        critic_params = [
            {
                "params": [
                    *list(self.critic.action_map_fc.parameters()),
                    *list(self.critic.layer1.parameters()),
                    *list(self.critic.layer2.parameters()),
                    *list(self.critic.layer3.parameters()),
                    *list(self.critic.value_head.parameters()),
                ],
                "lr": self.cnn_lr,
            },
            {
                "params": [
                    self.critic.cls_token,
                    self.critic.position_embedding,
                    *list(self.critic.cross_attention_layers.parameters()),
                    *list(self.critic.layer_norms.parameters()),
                ],
                "lr": self.other_lr,
            },
        ]

        self.actor_optimizer = Adam(actor_params, eps=1e-5)
        self.critic_optimizer = Adam(critic_params, eps=1e-5)

        self.critic_loss = torch.nn.MSELoss()

        self.scheduler = lambda step: max(1.0 - float(step / self.n_iter), 1e-5)

        def lr_lambda(step):
            if step < self.warmup_steps:
                return float(step) / float(max(1, self.warmup_steps))
            return max(0.1, float(self.warmup_steps**0.5) * (step**-0.5))

        self.actor_scheduler = LambdaLR(self.actor_optimizer, lr_lambda=lr_lambda)
        self.critic_scheduler = LambdaLR(self.critic_optimizer, lr_lambda=lr_lambda)

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
