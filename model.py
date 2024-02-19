import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import normal

from einops import rearrange


class Actor(nn.Module):
    def __init__(
        self, n_actions, image_size=32, hidden_dim=128, n_layers=3, num_heads=8
    ):
        super(Critic, self).__init__()
        self.hidden_dim = hidden_dim
        self.image_size = image_size
        self.n_layers = n_layers
        self.n_actions = n_actions

        # Define the size of patches
        self.patch_size = 4

        self.num_patches = (image_size // self.patch_size) ** 2

        # Embedding layers for action map
        self.action_map_fc = nn.Linear(3, hidden_dim)

        # Feature extractor layers
        self.feature_extractors = nn.ModuleList(
            [
                ResidualBlock(3, 32, stride=1),
                ResidualBlock(32, 64, stride=1),
                ResidualBlock(64, 128, stride=1),
            ]
        )
        self.num_patches_per_extractor = [
            image_size**2 // (ps**2) for ps in [4, 4, 4]
        ]  # 예를 들어 patch size가 모두 4라고 가정
        self.position_embeddings = nn.ParameterList(
            [
                nn.Parameter(torch.randn(1, num_patches, hidden_dim))
                for num_patches in self.num_patches_per_extractor
            ]
        )
        # cls token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        # Cross-attention layers
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
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)

        # Initialize the embeddings separately
        nn.init.normal_(self.cls_token, std=0.02)  # cls_token initialization
        nn.init.normal_(
            self.position_embedding, std=0.02
        )  # Position embedding initialization

    def forward(self, state, action_map):
        batch_size = state.size(0)
        # Embedding for action map
        action_map_emb = self.action_map_fc(action_map.view(-1, 3))
        action_map_emb = action_map_emb.view(batch_size, -1, self.hidden_dim)

        # cls_tokens preparation
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        total_patch_embeddings = []

        for i, extractor in enumerate(self.feature_extractors):
            feature = extractor(state)
            # Calculate the number of patches based on the feature map size and patch size
            num_patches = self.num_patches_per_extractor[i]
            # Extract patches and reshape for position embedding addition
            patches = rearrange(
                feature, "b c (h ph) (w pw) -> b (h w) (ph pw c)", ph=4, pw=4
            )
            patches = patches[
                :, :num_patches, :
            ]  # Ensure we only use the first 'num_patches' patches
            pos_embedding = self.position_embeddings[i].expand(batch_size, -1, -1)
            patches += pos_embedding
            total_patch_embeddings.append(patches)

        # Concatenate patch embeddings from different layers
        patch_embeddings = torch.cat(total_patch_embeddings, dim=1)
        action_map_emb = torch.cat((cls_tokens, action_map_emb), dim=1)

        # Apply cross-attention
        for layer_norm, attn_layer in zip(
            self.layer_norms, self.cross_attention_layers
        ):
            action_map_emb = layer_norm(action_map_emb)
            attn_output, _ = attn_layer(
                action_map_emb, patch_embeddings, patch_embeddings
            )
            action_map_emb += attn_output  # Apply residual connection

        # Extract cls token and apply value head
        cls_token_output = action_map_emb[:, 0, :]
        mu = self.mu(cls_token_output)
        std = torch.exp(self.log_std + 1e-5)

        dist = normal.Normal(mu, std)

        return dist


class Critic(nn.Module):
    def __init__(self, image_size=32, hidden_dim=128, n_layers=3, num_heads=8):
        super(Critic, self).__init__()
        self.hidden_dim = hidden_dim
        self.image_size = image_size
        self.n_layers = n_layers

        # Define the size of patches
        self.patch_size = 4

        self.num_patches = (image_size // self.patch_size) ** 2

        # Embedding layers for action map
        self.action_map_fc = nn.Linear(3, hidden_dim)

        # Feature extractor layers
        self.feature_extractors = nn.ModuleList(
            [
                ResidualBlock(3, 32, stride=1),
                ResidualBlock(32, 64, stride=1),
                ResidualBlock(64, 128, stride=1),
            ]
        )
        self.num_patches_per_extractor = [
            image_size**2 // (ps**2) for ps in [4, 4, 4]
        ]  # 예를 들어 patch size가 모두 4라고 가정
        self.position_embeddings = nn.ParameterList(
            [
                nn.Parameter(torch.randn(1, num_patches, hidden_dim))
                for num_patches in self.num_patches_per_extractor
            ]
        )
        # cls token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        # Cross-attention layers
        self.cross_attention_layers = nn.ModuleList(
            [nn.MultiheadAttention(hidden_dim, num_heads) for _ in range(n_layers)]
        )

        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(n_layers)]
        )

        # Value head
        self.value_head = nn.Linear(hidden_dim, 1)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, state, action_map):
        batch_size = state.size(0)

        # Embedding for action map
        action_map_emb = self.action_map_fc(action_map.view(-1, 3))
        action_map_emb = action_map_emb.view(batch_size, -1, self.hidden_dim)

        # cls_tokens preparation
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        total_patch_embeddings = []

        for i, extractor in enumerate(self.feature_extractors):
            feature = extractor(state)
            # Calculate the number of patches based on the feature map size and patch size
            num_patches = self.num_patches_per_extractor[i]
            # Extract patches and reshape for position embedding addition
            patches = rearrange(
                feature, "b c (h ph) (w pw) -> b (h w) (ph pw c)", ph=4, pw=4
            )
            patches = patches[
                :, :num_patches, :
            ]  # Ensure we only use the first 'num_patches' patches
            pos_embedding = self.position_embeddings[i].expand(batch_size, -1, -1)
            patches += pos_embedding
            total_patch_embeddings.append(patches)

        # Concatenate patch embeddings from different layers
        patch_embeddings = torch.cat(total_patch_embeddings, dim=1)
        action_map_emb = torch.cat((cls_tokens, action_map_emb), dim=1)

        # Apply cross-attention
        for layer_norm, attn_layer in zip(
            self.layer_norms, self.cross_attention_layers
        ):
            action_map_emb = layer_norm(action_map_emb)
            attn_output, _ = attn_layer(
                action_map_emb, patch_embeddings, patch_embeddings
            )
            action_map_emb += attn_output  # Apply residual connection

        # Extract cls token and apply value head
        cls_token_output = action_map_emb[:, 0, :]
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
