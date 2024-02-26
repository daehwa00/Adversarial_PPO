import torch
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision import transforms

import os
from collections import OrderedDict


# Load CIFAR10 dataset
transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=False, transform=transform
)
trainloader = DataLoader(trainset, batch_size=512, shuffle=False, num_workers=4)

testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=False, transform=transform
)
testloader = DataLoader(testset, batch_size=512, shuffle=False, num_workers=4)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


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


def load_model_and_filtered_loader(
    device, env_batch, model_path="./pre-trained models/cifar10_resnet18_best.pth"
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 모델 로딩과 초기화
    model = ResNet18().to(device)
    if model_path and os.path.isfile(model_path):
        state_dict = torch.load(model_path, map_location=device)["net"]
        model.load_state_dict(OrderedDict((k[7:], v) for k, v in state_dict.items()))
        print("Successfully loaded classifier model.")
    else:
        print("Model path is invalid. Exiting.")
        return None, None

    # 데이터셋으로부터 DataLoader 생성
    loader = DataLoader(trainset, batch_size=env_batch, shuffle=True, num_workers=4)

    # 정확도 평가 및 필터링된 데이터셋 생성
    accuracy, filtered_loader = evaluate_and_filter_dataset(
        loader, model, device, env_batch
    )

    return model, filtered_loader
