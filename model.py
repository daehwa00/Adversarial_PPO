import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import normal

from classifier import FeatureResNet18


class Actor(nn.Module):
    def __init__(
        self, n_actions, image_size=32, hidden_dim=512, n_layers=4, num_heads=8
    ):
        super(Actor, self).__init__()
        self.image_size = image_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_actions = n_actions
        self.patch_size = 8
        self.num_patches = 7**2

        self.action_map_fc = nn.Linear(3, hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.action_map_positional_embedding = nn.Parameter(
            torch.randn(image_size**2 + 1, hidden_dim)
        )

        self.patch_resnet = PatchResNet()
        self.position_embedding = nn.Parameter(
            torch.randn(self.num_patches, hidden_dim)
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

        action_map_emb = action_map_emb.permute(1, 0, 2)  # (seq, batch, feature)

        patches = self.patch_resnet(features)  # (batch, patches, feature)
        patches = patches + self.position_embedding.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        patches = patches.permute(1, 0, 2)  # (seq, batch, feature)

        for layer_norm, attn_layer in zip(
            self.layer_norms, self.cross_attention_layers
        ):
            action_map_emb = layer_norm(action_map_emb)
            attn_output, _ = attn_layer(action_map_emb, patches, patches)
            action_map_emb = action_map_emb + attn_output

        cls_token_output = action_map_emb[0, :, :]
        mu = self.mu(cls_token_output)
        std = torch.exp(self.log_std + 1e-5)

        dist = normal.Normal(mu, std)

        return dist

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.MultiheadAttention):
                nn.init.xavier_uniform_(m.in_proj_weight)
                nn.init.constant_(m.in_proj_bias, 0)
                nn.init.constant_(m.out_proj.bias, 0)
                nn.init.xavier_uniform_(m.out_proj.weight)

        # Special initializations for specific parameters
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.action_map_positional_embedding, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.02)
        nn.init.normal_(self.log_std, mean=0, std=0.1)


class Critic(nn.Module):
    def __init__(self, image_size=32, hidden_dim=512, n_layers=4, num_heads=8):
        super(Critic, self).__init__()
        self.image_size = image_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.patch_size = 8
        self.patch_stride = self.patch_size // 2
        self.num_patches = 7**2

        self.action_map_fc = nn.Linear(3, hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.action_map_positional_embedding = nn.Parameter(
            torch.randn(image_size**2 + 1, hidden_dim)
        )

        self.patch_resnet = PatchResNet()
        self.position_embedding = nn.Parameter(
            torch.randn(self.num_patches, hidden_dim)
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

        action_map_emb = action_map_emb.permute(1, 0, 2)  # (seq, batch, feature)

        patches = self.patch_resnet(features)  # (batch, patches, feature)
        patches = patches + self.position_embedding.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        patches = patches.permute(1, 0, 2)  # (seq, batch, feature)

        for layer_norm, attn_layer in zip(
            self.layer_norms, self.cross_attention_layers
        ):
            action_map_emb = layer_norm(action_map_emb)
            attn_output, _ = attn_layer(action_map_emb, patches, patches)
            action_map_emb = action_map_emb + attn_output

        cls_token_output = action_map_emb[0, :, :]

        value = self.value_head(cls_token_output)

        return value

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.MultiheadAttention):
                nn.init.xavier_uniform_(m.in_proj_weight)
                nn.init.constant_(m.in_proj_bias, 0)
                nn.init.constant_(m.out_proj.bias, 0)
                nn.init.xavier_uniform_(m.out_proj.weight)

        # Special initializations for specific parameters
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.action_map_positional_embedding, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.02)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.expansion = 1
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
    def __init__(self, block=ResidualBlock, patch_size=8, num_blocks=[2, 2, 2, 2]):
        super(PatchResNet, self).__init__()
        self.patch_size = patch_size
        self.patch_stride = self.patch_size // 2
        self.in_planes = 3

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        B, C, H, W = x.size()
        # 이미지를 패치로 나눔
        patches = x.unfold(2, self.patch_size, self.patch_stride).unfold(
            3, self.patch_size, self.patch_stride
        )
        # B, C, H, W -> B, C, Patches, Patch_size, Patch_size
        patches = patches.contiguous().view(B, C, -1, self.patch_size, self.patch_size)
        # B, C, Patches, Patch_size, Patch_size -> (B*Patches), C, Patch_size, Patch_size
        patches = (
            patches.permute(2, 0, 1, 3, 4)
            .contiguous()
            .view(-1, x.size(1), self.patch_size, self.patch_size)
        )

        # 모든 패치를 한 번에 처리
        patch_outputs = self.layer1(patches)
        patch_outputs = self.layer2(patch_outputs)
        patch_outputs = self.layer3(patch_outputs)
        patch_outputs = self.layer4(patch_outputs)

        # 결과 재구성
        # (B*Patches), C, H, W -> B, Patches, C, H, W
        out = patch_outputs.view(B, 49, -1)
        # B, Patches, C, H, W -> B, Patches, C*H*W (여기서는 예시로 C*H*W로 변환, 필요에 따라 조정 가능)

        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out
