import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import normal

from einops.layers.torch import Rearrange


class Actor(nn.Module):
    def __init__(
        self, n_actions, image_size=32, hidden_dim=128, n_layers=3, num_heads=4
    ):
        super(Actor, self).__init__()
        self.image_size = image_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.patch_size = 4
        self.num_patches = (image_size // self.patch_size) ** 2
        self.n_actions = n_actions

        self.action_map_fc = nn.Linear(3, hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.action_map_positional_embedding = nn.Parameter(
            torch.randn(image_size**2 + 1, hidden_dim)
        )

        # Positional embeddings for different feature map sizes
        self.position_embedding_1 = nn.Parameter(
            torch.randn(self.num_patches, hidden_dim)
        )
        self.position_embedding_2 = nn.Parameter(
            torch.randn(self.num_patches // 4, hidden_dim)
        )
        self.position_embedding_3 = nn.Parameter(
            torch.randn(self.num_patches // 16, hidden_dim)
        )

        # Feature extractors
        self.feature_extractor_1 = ResidualBlock(3, 64, stride=1)
        self.feature_extractor_2 = ResidualBlock(64, 128, stride=2)
        self.feature_extractor_3 = ResidualBlock(128, 256, stride=2)

        self.projection_1 = nn.Sequential(
            nn.Conv2d(
                64, self.hidden_dim, kernel_size=self.patch_size, stride=self.patch_size
            ),
            Rearrange("b c h w -> b (h w) c"),
        )
        # Projections for feature extractors
        self.projection_2 = nn.Sequential(
            nn.Conv2d(
                128, hidden_dim * 4, kernel_size=self.patch_size, stride=self.patch_size
            ),
            Rearrange("b (n c) h w -> b (n h w) c", n=4),
        )
        self.projection_3 = nn.Sequential(
            nn.Conv2d(
                256,
                hidden_dim * 16,
                kernel_size=self.patch_size,
                stride=self.patch_size,
            ),
            Rearrange("b (n c) h w -> b (n h w) c", n=16),
        )

        self.position_embedding = nn.Parameter(
            torch.randn((image_size // self.patch_size) ** 2, hidden_dim)
        )

        self.cross_attention_layers = nn.ModuleList(
            [nn.MultiheadAttention(hidden_dim, num_heads) for _ in range(n_layers)]
        )
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(n_layers)]
        )

        self.mu = nn.Linear(in_features=hidden_dim, out_features=n_actions)
        self.log_std = nn.Parameter(torch.zeros(1, self.n_actions))

        self._init_weights()

    def forward(self, features, action_map):
        batch_size = features.size(0)
        action_map_emb = self.action_map_fc(
            action_map.permute(0, 2, 3, 1).reshape(batch_size, -1, 3)
        )
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        action_map_emb = torch.cat(
            [cls_tokens, action_map_emb], dim=1
        ) + self.action_map_positional_embedding.unsqueeze(0).expand(batch_size, -1, -1)

        action_map_emb = action_map_emb.permute(1, 0, 2)

        # patch 1
        features_1 = self.feature_extractor_1(features)
        patches_1 = self.projection_1(features_1)
        pos_emb_1 = self.position_embedding_1.unsqueeze(0).expand(batch_size, -1, -1)
        patches_1 = patches_1 + pos_emb_1
        # patch 2
        features_2 = self.feature_extractor_2(features_1)
        patches_2 = self.projection_2(features_2)
        pos_emb_2 = (
            self.position_embedding_2.unsqueeze(0)
            .repeat(4, 1, 1)
            .view(64, self.hidden_dim)
        )
        pos_emb_2 = pos_emb_2.unsqueeze(0).expand(batch_size, -1, -1)
        patches_2 = patches_2 + pos_emb_2
        # patch 3
        features_3 = self.feature_extractor_3(features_2)
        patches_3 = self.projection_3(features_3)
        pos_emb_3 = (
            self.position_embedding_3.unsqueeze(0)
            .repeat(16, 1, 1)
            .view(64, self.hidden_dim)
        )
        pos_emb_3 = pos_emb_3.unsqueeze(0).expand(batch_size, -1, -1)
        patches_3 = patches_3 + pos_emb_3
        # Combine all patch embeddings
        patch_embeddings = torch.cat([patches_1, patches_2, patches_3], dim=1).permute(
            1, 0, 2
        )

        for layer_norm, attn_layer in zip(
            self.layer_norms, self.cross_attention_layers
        ):
            action_map_emb = layer_norm(action_map_emb)
            attn_output, _ = attn_layer(
                action_map_emb, patch_embeddings, patch_embeddings
            )
            action_map_emb = action_map_emb + attn_output

        cls_token_output = action_map_emb[0, :, :]
        mu = self.mu(cls_token_output)
        std = torch.exp(self.log_std + 1e-5)

        dist = normal.Normal(mu, std)

        return dist

    def _init_weights(self):
        # Initialize linear layers using Xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

            # Initialize convolutional layers using Kaiming initialization
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            # Initialize parameters for multi-head attention layers
            elif isinstance(m, nn.MultiheadAttention):
                for param in m.parameters():
                    if param.dim() > 1:
                        nn.init.xavier_uniform_(param)

        # Initialize embeddings and other parameters with a normal distribution
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.normal_(self.action_map_positional_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding_1, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding_2, mean=0.0, std=0.02)
        nn.init.normal_(self.log_std, mean=0.0, std=0.02)


class Critic(nn.Module):
    def __init__(self, image_size=32, hidden_dim=128, n_layers=3, num_heads=4):
        super(Critic, self).__init__()

        self.image_size = image_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.patch_size = 4
        self.num_patches = (image_size // self.patch_size) ** 2

        self.action_map_fc = nn.Linear(3, hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.action_map_positional_embedding = nn.Parameter(
            torch.randn(image_size**2 + 1, hidden_dim)
        )

        # Positional embeddings for different feature map sizes
        self.position_embedding_1 = nn.Parameter(
            torch.randn(self.num_patches, hidden_dim)
        )
        self.position_embedding_2 = nn.Parameter(
            torch.randn(self.num_patches // 4, hidden_dim)
        )
        self.position_embedding_3 = nn.Parameter(
            torch.randn(self.num_patches // 16, hidden_dim)
        )

        # Feature extractors
        self.feature_extractor_1 = ResidualBlock(3, 64, stride=1)
        self.feature_extractor_2 = ResidualBlock(64, 128, stride=2)
        self.feature_extractor_3 = ResidualBlock(128, 256, stride=2)

        self.projection_1 = nn.Sequential(
            nn.Conv2d(
                64, self.hidden_dim, kernel_size=self.patch_size, stride=self.patch_size
            ),
            Rearrange("b c h w -> b (h w) c"),
        )
        # Projections for feature extractors
        self.projection_2 = nn.Sequential(
            nn.Conv2d(
                128, hidden_dim * 4, kernel_size=self.patch_size, stride=self.patch_size
            ),
            Rearrange("b (n c) h w -> b (n h w) c", n=4),
        )
        self.projection_3 = nn.Sequential(
            nn.Conv2d(
                256,
                hidden_dim * 16,
                kernel_size=self.patch_size,
                stride=self.patch_size,
            ),
            Rearrange("b (n c) h w -> b (n h w) c", n=16),
        )

        self.position_embedding = nn.Parameter(
            torch.randn((image_size // self.patch_size) ** 2, hidden_dim)
        )

        self.cross_attention_layers = nn.ModuleList(
            [nn.MultiheadAttention(hidden_dim, num_heads) for _ in range(n_layers)]
        )
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(n_layers)]
        )

        self.value_head = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def forward(self, features, action_map):
        batch_size = features.size(0)
        action_map_emb = self.action_map_fc(
            action_map.permute(0, 2, 3, 1).reshape(batch_size, -1, 3)
        )
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        action_map_emb = torch.cat(
            [cls_tokens, action_map_emb], dim=1
        ) + self.action_map_positional_embedding.unsqueeze(0).expand(batch_size, -1, -1)

        action_map_emb = action_map_emb.permute(1, 0, 2)

        # patch 1
        features_1 = self.feature_extractor_1(features)
        patches_1 = self.projection_1(features_1)
        pos_emb_1 = self.position_embedding_1.unsqueeze(0).expand(batch_size, -1, -1)
        patches_1 = patches_1 + pos_emb_1
        # patch 2
        features_2 = self.feature_extractor_2(features_1)
        patches_2 = self.projection_2(features_2)
        pos_emb_2 = (
            self.position_embedding_2.unsqueeze(0)
            .repeat(4, 1, 1)
            .view(64, self.hidden_dim)
        )
        pos_emb_2 = pos_emb_2.unsqueeze(0).expand(batch_size, -1, -1)
        patches_2 = patches_2 + pos_emb_2
        # patch 3
        features_3 = self.feature_extractor_3(features_2)
        patches_3 = self.projection_3(features_3)
        pos_emb_3 = (
            self.position_embedding_3.unsqueeze(0)
            .repeat(16, 1, 1)
            .view(64, self.hidden_dim)
        )
        pos_emb_3 = pos_emb_3.unsqueeze(0).expand(batch_size, -1, -1)
        patches_3 = patches_3 + pos_emb_3
        # Combine all patch embeddings
        patch_embeddings = torch.cat([patches_1, patches_2, patches_3], dim=1).permute(
            1, 0, 2
        )

        for layer_norm, attn_layer in zip(
            self.layer_norms, self.cross_attention_layers
        ):
            action_map_emb = layer_norm(action_map_emb)
            attn_output, _ = attn_layer(
                action_map_emb, patch_embeddings, patch_embeddings
            )
            action_map_emb = action_map_emb + attn_output

        cls_token_output = action_map_emb[0, :, :]
        value = self.value_head(cls_token_output)

        return value

    def _init_weights(self):
        # Initialize linear layers using Xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

            # Initialize convolutional layers using Kaiming initialization
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            # Initialize parameters for multi-head attention layers
            elif isinstance(m, nn.MultiheadAttention):
                for param in m.parameters():
                    if param.dim() > 1:
                        nn.init.xavier_uniform_(param)

        # Initialize embeddings and other parameters with a normal distribution
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.normal_(self.action_map_positional_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding_1, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding_2, mean=0.0, std=0.02)


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


class PatchResNet(nn.Module):
    def __init__(self, patch_size=8, in_channels=3):
        super(PatchResNet, self).__init__()
        self.patch_size = patch_size
        self.res_blocks = nn.Sequential(
            ResidualBlock(in_channels, 64, stride=1),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 512, stride=2),
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        # 이미지를 패치로 나눔
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(
            3, self.patch_size, self.patch_size
        )
        patches = patches.contiguous().view(-1, 3, self.patch_size, self.patch_size)

        # 각 패치에 대해 ResNet 구조 적용
        batch_size, C, H, W = patches.size()
        patches = patches.view(-1, C, H, W)
        res_out = self.res_blocks(patches)

        # 패치별 평균 풀링 적용
        out = self.adaptive_pool(res_out)
        out = out.view(batch_size, -1, 512)

        # 패치 결과 병합 (예시에서는 단순히 리턴, 실제 사용 시 조정 필요)
        return out
