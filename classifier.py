import torch
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import os
from collections import OrderedDict


# Load CIFAR10 dataset
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

dataset = datasets.ImageFolder("path/to/imagenet/validation", transform=transform)

# CIFAR10 클래스
classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def evaluate_and_filter_dataset(loader, model, device, env_batch):
    model.eval()  # 모델을 평가 모드로 설정
    correct_indices = []  # 정확하게 분류된 데이터의 인덱스를 저장할 리스트
    total, correct = 0, 0  # 총 데이터 수와 정확하게 분류된 데이터 수

    with torch.no_grad():  # 기울기 계산을 비활성화
        for i, data in enumerate(loader, 0):
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            correct_batch_indices = (
                (predicted == labels).nonzero(as_tuple=False).squeeze().tolist()
            )
            # 배치 내에서 정확하게 분류된 데이터의 인덱스를 전체 데이터셋에 대한 인덱스로 변환하여 추가
            correct_indices.extend(
                [
                    i * loader.batch_size + idx
                    for idx in correct_batch_indices
                    if isinstance(idx, int) or idx >= 0
                ]
            )

    accuracy = 100.0 * correct / total  # 정확도 계산
    print(f"Test Accuracy: {accuracy:.2f}%")

    # env_batch로 나누어 떨어지도록 인덱스를 조정
    num_complete_batches = len(correct_indices) // env_batch
    filtered_indices = correct_indices[: num_complete_batches * env_batch]

    # 정확하게 분류된 데이터만 포함하는 새로운 데이터셋 생성
    filtered_dataset = Subset(loader.dataset, filtered_indices)
    filtered_loader = DataLoader(
        filtered_dataset, batch_size=env_batch, shuffle=True, num_workers=4
    )
    print(f"Filtered Dataset Size: {len(filtered_dataset)}")

    return accuracy, filtered_loader


def load_model_and_filtered_loader(device, env_batch):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 모델 로딩과 초기화
    model = models.ResNeXt101_32X8D_Weights(pretrained=True)
    # 데이터셋으로부터 DataLoader 생성
    loader = DataLoader(dataset, batch_size=env_batch, shuffle=True, num_workers=4)

    # 정확도 평가 및 필터링된 데이터셋 생성
    accuracy, filtered_loader = evaluate_and_filter_dataset(
        loader, model, device, env_batch
    )

    return model, filtered_loader
