from agent import Agent
import torch
from classifier import load_model_and_filtered_loader
from autoencoder import load_pretrained_autoencoder
from environment import make_env
from train import Train
from utils import set_random_seed

ENV_NAME = "AdversarialRL"


lr = 1e-4
epochs = 10
clip_range = 0.2
mini_batch_size = 200
num_heads = 4
T = 64  # Horizon
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

env_batch = 200
hidden_dim = 128
action_map_size = [3, 32, 32]
n_layers = 4
warmup_steps = 200
n_iterations = warmup_steps * 10
# Reward weights
alpha = 0
beta = 5.0
gamma = 0.03

if __name__ == "__main__":
    set_random_seed(2024, deterministic=True)
    pre_trained_classifier, filtered_loader = load_model_and_filtered_loader(
        device, env_batch
    )
    env = make_env(
        classifier=pre_trained_classifier,
        filtered_loader=filtered_loader,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )

    agent = Agent(
        env_name=ENV_NAME,
        n_iter=n_iterations,
        n_actions=env.action_space,
        action_map_size=action_map_size,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        num_heads=num_heads,
        lr=lr,
        warmup_steps=warmup_steps,
    )

    trainer = Train(
        env=env,
        env_name=ENV_NAME,
        agent=agent,
        horizon=T,
        n_iterations=n_iterations,
        epochs=epochs,
        mini_batch_size=mini_batch_size,
        epsilon=clip_range,
    )
    trainer.step()
