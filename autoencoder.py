import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from piqa import SSIM
import numpy as np
import matplotlib
import os

matplotlib.use("TkAgg")  # TkAgg 백엔드로 변경


# AutoEncoder 모델 정의
class AutoEncoder(nn.Module):
    def __init__(self, latent_size=512):
        super(AutoEncoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # [16, 16, 16]
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # [32, 8, 8]
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # [64, 4, 4]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # [128, 2, 2]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, latent_size, 2),  # [latent_size, 1, 1]
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_size, 128, 2),  # [128, 2, 2]
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(
                128, 64, 3, stride=2, padding=1, output_padding=1
            ),  # [64, 4, 4]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(
                64, 32, 3, stride=2, padding=1, output_padding=1
            ),  # [32, 8, 8]
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(
                32, 16, 3, stride=2, padding=1, output_padding=1
            ),  # [16, 16, 16]
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(
                16, 3, 3, stride=2, padding=1, output_padding=1
            ),  # [3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def load_pretrained_autoencoder(
    model_path="./pre-trained models/best_AE_model.pth", latent_size=512
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoEncoder(latent_size=latent_size).to(device)

    # 모델 상태를 불러오기 전에 파일 존재 여부 확인
    if os.path.isfile(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print("Model loaded successfully.")
        except Exception as e:
            print(f"An error occurred while loading the model: {e}")
            print("Initializing a new model.")
    else:
        print(f"No saved model found at {model_path}. Initializing a new model.")

    return model
