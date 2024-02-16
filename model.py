import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import normal


class Actor(nn.Module):
    def __init__(self, n_actions, image_size, hidden_dim=256):
        super(Actor, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.hidden_dim = hidden_dim
        self.image_size = image_size

        # Action map 임베딩을 위한 FC 레이어
        self.action_map_fc = nn.Linear(3, hidden_dim)  # RGB 3차원 -> hidden_dim 임베딩

        # CNN feature extractor 구성
        self.layer1 = ResidualBlock(3, 32, stride=1)
        self.layer2 = ResidualBlock(32, 64, stride=2)
        self.layer3 = ResidualBlock(64, hidden_dim, stride=2)

        # cls_token 임베딩
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        # 위치 임베딩
        self.position_embedding = nn.Parameter(
            torch.randn(1, image_size * image_size + 1, hidden_dim)
        )
        # Cross-attention layer
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8)

        self.mu = nn.Linear(in_features=hidden_dim, out_features=n_actions)
        self.log_std = nn.Parameter(torch.zeros(1, self.n_actions))

        self._init_weights()

    def _init_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                layer.bias.data.zero_()

    def forward(self, state, action_map):
        batch_size = state.size(0)

        # Action map 임베딩 및 위치 임베딩 추가
        action_map_emb = self.action_map_fc(action_map.view(-1, 3))
        action_map_emb = action_map_emb.view(
            batch_size, self.image_size * self.image_size, self.hidden_dim
        )
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        action_map_emb = (
            torch.cat((cls_tokens, action_map_emb), dim=1) + self.position_embedding
        )

        # CNN을 통한 특징 추출
        cnn_out = self.layer1(state)
        cnn_out = self.layer2(cnn_out)
        cnn_out = self.layer3(cnn_out)

        # CNN feature map을 sequence of vectors로 변환
        cnn_out = cnn_out.permute(0, 2, 3, 1).contiguous()  # (batch_size, H, W, C)
        cnn_out = cnn_out.view(batch_size, -1, self.hidden_dim)  # (batch_size, H*W, C)

        # Cross-attention: CNN features를 key와 value로, action map 임베딩을 query로 사용
        attn_output, _ = self.cross_attention(
            query=action_map_emb.permute(1, 0, 2),  # (seq_len, batch_size, hidden_dim)
            key=cnn_out.permute(1, 0, 2),  # (seq_len, batch_size, hidden_dim)
            value=cnn_out.permute(1, 0, 2),  # (seq_len, batch_size, hidden_dim)
        )

        # 최종 가치 예측을 위해 cls_token만 사용
        cls_token_output = attn_output[0]
        mu = self.mu(cls_token_output)
        std = torch.exp(self.log_std + 1e-5)

        dist = normal.Normal(mu, std)

        return dist


class Critic(nn.Module):
    def __init__(self, hidden_dim=128, image_size=32):
        super(Critic, self).__init__()
        self.hidden_dim = hidden_dim
        self.image_size = image_size

        # Action map 임베딩을 위한 FC 레이어
        self.action_map_fc = nn.Linear(3, hidden_dim)  # RGB 3차원 -> hidden_dim 임베딩

        # CNN feature extractor 구성
        self.layer1 = ResidualBlock(3, 32, stride=1)
        self.layer2 = ResidualBlock(32, 64, stride=2)
        self.layer3 = ResidualBlock(64, hidden_dim, stride=2)

        # cls_token 임베딩
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        # 위치 임베딩
        self.position_embedding = nn.Parameter(
            torch.randn(1, image_size * image_size + 1, hidden_dim)
        )
        # Cross-attention layer
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8)
        # 최종 가치를 예측하기 위한 레이어
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, state, action_map):
        batch_size = state.size(0)

        # Action map 임베딩 및 위치 임베딩 추가
        action_map_emb = self.action_map_fc(action_map.view(-1, 3))
        action_map_emb = action_map_emb.view(
            batch_size, self.image_size * self.image_size, self.hidden_dim
        )
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        action_map_emb = (
            torch.cat((cls_tokens, action_map_emb), dim=1) + self.position_embedding
        )

        # CNN을 통한 특징 추출
        cnn_out = self.layer1(state)
        cnn_out = self.layer2(cnn_out)
        cnn_out = self.layer3(cnn_out)

        # CNN feature map을 sequence of vectors로 변환
        cnn_out = cnn_out.permute(0, 2, 3, 1).contiguous()  # (batch_size, H, W, C)
        cnn_out = cnn_out.view(batch_size, -1, self.hidden_dim)  # (batch_size, H*W, C)

        # Cross-attention: CNN features를 key와 value로, action map 임베딩을 query로 사용
        attn_output, _ = self.cross_attention(
            query=action_map_emb.permute(1, 0, 2),  # (seq_len, batch_size, hidden_dim)
            key=cnn_out.permute(1, 0, 2),  # (seq_len, batch_size, hidden_dim)
            value=cnn_out.permute(1, 0, 2),  # (seq_len, batch_size, hidden_dim)
        )

        # 최종 가치 예측을 위해 cls_token만 사용
        cls_token_output = attn_output[0]
        value = self.value_head(cls_token_output)

        return value


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
