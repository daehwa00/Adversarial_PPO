import numpy as np
import torch


class Env:
    def __init__(
        self,
        classifier,
        filtered_loader,
        latent_vector_size,
        time_horizon=10,
    ):
        self.classifier = classifier
        self.original_loader = filtered_loader
        self.filtered_loader = iter(self.original_loader)
        self.time_horizon = time_horizon
        self.state_space = latent_vector_size  # latent vector size
        self.action_space = 5  # R, G, B, X, Y
        self.env_batch = filtered_loader.batch_size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        self.original_class = torch.argmax(self.original_prediction, dim=1)

        return self.current_state

    def step(self, actions):
        modified_image = self.apply_actions(self.current_state, actions)

        # 수정된 이미지를 모델과 동일한 장치로 이동
        modified_image = modified_image.to(self.device)
        new_prediction = self.classifier(modified_image)

        reward = self.calculate_reward(new_prediction)
        argmax_new_prediction = torch.argmax(new_prediction, dim=1)

        done = self.original_class != argmax_new_prediction
        done = done.float()

        self.current_state = modified_image

        # show modified image
        # plt.imshow(modified_image[0].cpu().numpy().transpose(1, 2, 0))
        # plt.show()

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
        # X 를 int로 변환
        X = X.astype(int)
        # Y 를 int로 변환
        Y = Y.astype(int)
        # 각 이미지에 대해 RGB 값을 적용
        batch_indices = np.arange(images.shape[0])
        images[batch_indices, :, Y, X] = np.stack([R, G, B], axis=-1)

        # NumPy 배열을 텐서로 변환
        modified_images = torch.from_numpy(images).float().to(self.device)

        return modified_images

    def calculate_reward(self, classifier_results):
        with torch.no_grad():
            classifier_results = torch.softmax(classifier_results, dim=1)
            batch_indices = torch.arange(classifier_results.size(0)).to(self.device)
            selected_probs = classifier_results[batch_indices, self.original_class]
            safe_values = torch.clamp(1 - selected_probs, min=1e-8, max=1.0 - 1e-5)
            reward = torch.log10(safe_values).cpu()

        return reward


def make_env(classifier, filtered_loader, latent_vector_size=512):
    return Env(classifier, filtered_loader, latent_vector_size)
