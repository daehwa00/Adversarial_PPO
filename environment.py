import numpy as np
import torch


class Env:
    def __init__(
        self,
        classifier,
        filtered_loader,
        latent_vector_size,
        time_horizon=10,
        alpha=1.0,  # 점진적 보상 가중치
        beta=20.0,  # 최종 보상 가중치
        gamma=0.01,  # 효율성 보상 가중치
    ):
        self.classifier = classifier
        self.original_loader = filtered_loader
        self.filtered_loader = iter(self.original_loader)
        self.time_horizon = time_horizon
        self.state_space = latent_vector_size  # latent vector size
        self.action_space = 5  # R, G, B, X, Y
        self.env_batch = filtered_loader.batch_size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.actions_taken = 0

    def reset(self):
        while True:
            try:
                self.current_state, _ = next(self.filtered_loader)
                break
            except StopIteration:
                self.filtered_loader = iter(
                    self.original_loader
                )  # 새로운 이터레이터 생성

        # 입력 데이터를 모델과 동일한 장치로 이동
        self.current_state = self.current_state.to(self.device)
        self.original_prediction = torch.softmax(
            self.classifier(self.current_state), dim=1
        )
        self.previous_prediction = self.original_prediction
        self.original_class = torch.argmax(self.original_prediction, dim=1)
        self.actions_taken = 0
        return self.current_state

    def step(self, actions):
        self.actions_taken += 1
        modified_image = self.apply_actions(self.current_state, actions)

        # 수정된 이미지를 모델과 동일한 장치로 이동
        modified_image = modified_image.to(self.device)
        new_prediction = self.classifier(modified_image)

        reward = self.calculate_reward(new_prediction)
        argmax_new_prediction = torch.argmax(new_prediction, dim=1)

        done = (self.original_class != argmax_new_prediction).float()

        # done이면 추가 보상 50을 줌
        reward += done.cpu() * self.beta

        self.current_state = modified_image
        self.previous_prediction = new_prediction

        return modified_image, reward, done, {}

    def apply_actions(self, images, actions):
        # action 텐서를 CPU로 이동시키고 NumPy 배열로 변환
        if torch.is_tensor(actions):
            actions = actions.cpu().numpy()
        if torch.is_tensor(images):
            images = images.cpu().numpy()
        R, G, B, X, Y = (
            actions[:, 0],
            actions[:, 1],
            actions[:, 2],
            actions[:, 3],
            actions[:, 4],
        )
        X = X.astype(int)
        Y = Y.astype(int)
        batch_indices = np.arange(images.shape[0])
        images[batch_indices, :, Y, X] = np.stack([R, G, B], axis=-1)

        # NumPy 배열을 텐서로 변환
        modified_images = torch.from_numpy(images).float().to(self.device)

        return modified_images

    def calculate_reward(self, new_prediction):
        with torch.no_grad():
            new_prediction = torch.softmax(new_prediction, dim=1)
            batch_indices = torch.arange(new_prediction.size(0)).to(self.device)
            new_probs = new_prediction[batch_indices, self.original_class]
            prev_probs = self.previous_prediction[batch_indices, self.original_class]
            reward = self.alpha * (new_probs - prev_probs).cpu()
            reward = torch.where(reward > 0, reward, torch.zeros_like(reward))

            efficiency_reward = -self.gamma * self.actions_taken
            reward += efficiency_reward

        return reward.cpu()


def make_env(
    classifier,
    filtered_loader,
    latent_vector_size=512,
    alpha=1.0,
    beta=50.0,
    gamma=0.01,
):
    return Env(
        classifier,
        filtered_loader,
        latent_vector_size,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )
