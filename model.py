import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import normal

from einops import rearrange
from einops.layers.torch import Rearrange
import math


class Actor(nn.Module):
    def __init__(
        self, n_actions, image_size=32, hidden_dim=64, n_layers=3, num_heads=4
    ):
        super(Actor, self).__init__()
        self.image_size = image_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.patch_size = 4  # Assume square patches
        self.n_actions = n_actions

        # Define patch sizes and their corresponding positional embeddings
        self.num_patches = (image_size // self.patch_size) ** 2

        self.action_map_fc = nn.Linear(3, hidden_dim)
        self.projection = nn.Sequential(
            nn.Conv2d(
                3, self.hidden_dim, kernel_size=self.patch_size, stride=self.patch_size
            ),
            Rearrange("b c h w -> b (h w) c"),
        )
        self.feature_extractors = nn.ModuleList(
            [
                ResidualBlock(3, 32, stride=2),
                ResidualBlock(32, 64, stride=2),
            ]
        )
        self.projections = nn.ModuleList(
            [
                nn.Conv2d(
                    32,
                    self.hidden_dim,
                    kernel_size=self.patch_size,
                    stride=self.patch_size,
                ),
                nn.Conv2d(
                    64,
                    self.hidden_dim,
                    kernel_size=self.patch_size,
                    stride=self.patch_size,
                ),
            ]
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        self.position_embedding = nn.Parameter(
            torch.randn((image_size // self.patch_size) ** 2, hidden_dim)
        )
        self.position_embeddings = nn.ParameterList(
            [
                nn.Parameter(torch.randn(16, hidden_dim)),
                nn.Parameter(torch.randn(4, hidden_dim)),
            ]
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

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, features, action_map):
        batch_size = features.size(0)
        action_map_emb = self.action_map_fc(
            action_map.permute(0, 2, 3, 1).reshape(batch_size, -1, 3)
        )
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        patch_embeddings = []

        patches = self.projection(features)
        patch_embeddings.append(patches)

        for i, (extractor, projection, pos_embedding) in enumerate(
            zip(
                self.feature_extractors,
                self.projections,
                self.position_embeddings,
            )
        ):
            features = extractor(features)
            patches = projection(features)
            b, c, h, w = patches.size()
            patches = rearrange(patches, "b c (h) (w) -> b (h w) c", h=h, w=w)
            pos_embedding = pos_embedding.unsqueeze(0).expand(batch_size, -1, -1)

            patches = patches + pos_embedding

            patch_embeddings.append(patches)

        patch_embeddings = torch.cat(patch_embeddings, dim=1).permute(1, 0, 2)
        action_map_emb = torch.cat([cls_tokens, action_map_emb], dim=1).permute(1, 0, 2)

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


class Critic(nn.Module):
    def __init__(self, image_size=32, hidden_dim=128, n_layers=3, num_heads=8):
        super(Critic, self).__init__()
        self.image_size = image_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.patch_size = 4  # Assume square patches
        # Define patch sizes and their corresponding positional embeddings
        self.num_patches = (image_size // self.patch_size) ** 2

        self.action_map_fc = nn.Linear(3, hidden_dim)
        self.projection = nn.Sequential(
            nn.Conv2d(
                3, self.hidden_dim, kernel_size=self.patch_size, stride=self.patch_size
            ),
            Rearrange("b c h w -> b (h w) c"),
        )
        self.feature_extractors = nn.ModuleList(
            [
                ResidualBlock(3, 32, stride=2),
                ResidualBlock(32, 64, stride=2),
            ]
        )
        self.projections = nn.ModuleList(
            [
                nn.Conv2d(
                    32,
                    self.hidden_dim,
                    kernel_size=self.patch_size,
                    stride=self.patch_size,
                ),
                nn.Conv2d(
                    64,
                    self.hidden_dim,
                    kernel_size=self.patch_size,
                    stride=self.patch_size,
                ),
            ]
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        self.position_embedding = nn.Parameter(
            torch.randn((image_size // self.patch_size) ** 2, hidden_dim)
        )
        self.position_embeddings = nn.ParameterList(
            [
                nn.Parameter(torch.randn(16, hidden_dim)),
                nn.Parameter(torch.randn(4, hidden_dim)),
            ]
        )
        self.cross_attention_layers = nn.ModuleList(
            [nn.MultiheadAttention(hidden_dim, num_heads) for _ in range(n_layers)]
        )
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(n_layers)]
        )

        self.value_head = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, features, action_map):
        batch_size = features.size(0)
        action_map_emb = self.action_map_fc(
            action_map.permute(0, 2, 3, 1).reshape(batch_size, -1, 3)
        )
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        patch_embeddings = []

        patches = self.projection(features)
        patch_embeddings.append(patches)

        for i, (extractor, projection, pos_embedding) in enumerate(
            zip(
                self.feature_extractors,
                self.projections,
                self.position_embeddings,
            )
        ):
            features = extractor(features)
            patches = projection(features)
            b, c, h, w = patches.size()
            patches = rearrange(patches, "b c (h) (w) -> b (h w) c", h=h, w=w)
            pos_embedding = pos_embedding.unsqueeze(0).expand(batch_size, -1, -1)

            patches = patches + pos_embedding

            patch_embeddings.append(patches)

        patch_embeddings = torch.cat(patch_embeddings, dim=1).permute(1, 0, 2)
        action_map_emb = torch.cat([cls_tokens, action_map_emb], dim=1).permute(1, 0, 2)

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
