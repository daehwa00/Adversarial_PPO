from agent import Agent
import torch
from classifier import load_model_and_filtered_loader
from autoencoder import load_pretrained_autoencoder
from environment import make_env
from train import Train
from utils import set_random_seed

ENV_NAME = "AdversarialRL"


n_iterations = 5000
lr = 0.0005
epochs = 10
clip_range = 0.2
mini_batch_size = 128
T = 64
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
latent_size = 512
env_batch = 128
hidden_dim = 128
action_map_size = [3, 32, 32]
n_layers = 3

# Reward weights
alpha = 1.0
beta = 20.0
gamma = 0.1

if __name__ == "__main__":
    set_random_seed(2024, deterministic=True)
    pre_trained_classifier, filtered_loader = load_model_and_filtered_loader(
        device, env_batch
    )
    env = make_env(
        classifier=pre_trained_classifier,
        filtered_loader=filtered_loader,
        latent_vector_size=latent_size,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )

    agent = Agent(
        env_name=ENV_NAME,
        n_iter=n_iterations,
        n_states=env.state_space,
        n_actions=env.action_space,
        action_map_size=action_map_size,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        lr=lr,
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
