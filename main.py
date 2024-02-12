from agent import EncodedAgent
import torch
from classifier import load_model_and_filtered_loader
from autoencoder import load_pretrained_autoencoder
from environment import make_env
from train import Train
from utils import set_random_seed

ENV_NAME = "AdversarialRL"


n_iterations = 50000
lr = 0.0005
epochs = 10
clip_range = 0.2
mini_batch_size = 128
T = 64
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
latent_size = 512
env_batch = 256

# Reward weights
alpha = 1.0
beta = 30.0
gamma = 0.05

if __name__ == "__main__":
    set_random_seed(2024, deterministic=True)
    pre_trained_classifier, filtered_loader = load_model_and_filtered_loader(
        device, env_batch
    )
    autoencoder = load_pretrained_autoencoder(latent_size=latent_size)
    encoder = autoencoder.encoder
    env = make_env(
        classifier=pre_trained_classifier,
        filtered_loader=filtered_loader,
        latent_vector_size=latent_size,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )

    agent = EncodedAgent(
        encoder=encoder,
        env_name=ENV_NAME,
        n_iter=n_iterations,
        n_states=env.state_space,
        n_actions=env.action_space,
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
