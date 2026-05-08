import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def load_checkpoint_compat(path, map_location):
    # PyTorch >=2.6 defaults to weights_only=True.
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except Exception:
        return torch.load(path, map_location=map_location, weights_only=False)


class PCNN(nn.Module):
    """Perception CNN for facial expression recognition.

    This follows the public PCNN repository structure: six ResNet-18 branches
    are initialized from `models/resnet18_msceleb.pth`, five local regions are
    fused with the global face feature, and the network returns the main logits
    plus the summed local-branch logits.
    """

    def __init__(
        self,
        num_class=7,
        device="cpu",
        backbone_path="models/resnet18_msceleb.pth",
        require_backbone=True,
        use_local_enhancement=False,
        use_bidirectional_interaction=False,
        use_gated_fusion=False,
        use_gapw=False,
        gapw_patch_grid=4,
        gapw_temperature=1.0,
    ):
        super().__init__()

        self.resnet = models.resnet18()
        self.resnet1 = models.resnet18()
        self.resnet2 = models.resnet18()
        self.resnet3 = models.resnet18()
        self.resnet4 = models.resnet18()
        self.resnet5 = models.resnet18()
        self.resnet6 = models.resnet18()

        if os.path.exists(backbone_path):
            checkpoint = load_checkpoint_compat(backbone_path, map_location=device)
            state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
            for net in (
                self.resnet,
                self.resnet1,
                self.resnet2,
                self.resnet3,
                self.resnet4,
                self.resnet5,
                self.resnet6,
            ):
                net.load_state_dict(state_dict, strict=True)
        elif require_backbone:
            raise FileNotFoundError(
                f"Missing backbone weights: {backbone_path}. "
                "Download resnet18_msceleb.pth from the author's Google Drive "
                "and place it under models/."
            )

        self.features1 = nn.Sequential(*list(self.resnet.children())[:-3])
        self.features2 = nn.Sequential(*list(self.resnet1.children())[:-3])
        self.features3 = nn.Sequential(*list(self.resnet2.children())[:-3])
        self.features4 = nn.Sequential(*list(self.resnet3.children())[:-3])
        self.features6 = nn.Sequential(*list(self.resnet5.children())[:-3])
        self.features7 = nn.Sequential(*list(self.resnet6.children())[:-3])
        self.features8 = nn.Sequential(*list(self.resnet.children())[-3:-2])

        self.w1 = 0.5
        self.w2 = 0.75
        self.h1 = 0.5
        self.h2 = 0.65

        self.fc_loc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 32),
            nn.ReLU(),
            nn.Linear(32, 3 * 2),
        )
        self.use_local_enhancement = use_local_enhancement
        self.register_buffer(
            "identity_theta",
            torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).view(1, 2, 3),
        )
        self.stn_delta_scale = 0.1

        self.fc = nn.Linear(512, num_class)
        self.fc2 = nn.Linear(256, num_class)
        self.fc3 = nn.Linear(256, num_class)
        self.fc4 = nn.Linear(256, num_class)
        self.fc6 = nn.Linear(256, num_class)
        self.fc7 = nn.Linear(256, num_class)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.use_gated_fusion = use_gated_fusion
        self.gated_fusion = GatedFusionHead(512, 256, num_class) if use_gated_fusion else None
        self.use_gapw = use_gapw
        self.gapw = (
            GlobalGuidedPatchWeighting(
                256,
                patch_grid=gapw_patch_grid,
                temperature=gapw_temperature,
            )
            if use_gapw
            else None
        )
        self.local_enhancement = (
            LocalEnhancementBlock(256) if use_local_enhancement else nn.Identity()
        )
        self.use_bidirectional_interaction = use_bidirectional_interaction
        self.bidirectional_interaction = (
            BidirectionalInteractionBlock(256) if use_bidirectional_interaction else None
        )

        if self.use_local_enhancement:
            self._init_stn_as_identity()

    def _init_stn_as_identity(self):
        final = self.fc_loc[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, x):
        width = x.size(3)
        height = x.size(2)
        w_1 = int(width * self.w1)
        h_1 = int(height * self.h1)
        h_2 = int(height * self.h2)

        x2 = x[:, :, 0:h_1, 0:w_1]
        x3 = x[:, :, 0:h_1, w_1:width]
        x4 = x[:, :, h_1:h_2, 0:w_1]
        x6 = x[:, :, h_1:h_2, w_1:width]
        x7 = x[:, :, h_2:height, :]

        x1 = self.features1(x)
        x2 = self.features2(x2)
        x3 = self.features3(x3)
        x4 = self.features4(x4)
        x6 = self.features6(x6)
        x7 = self.features7(x7)

        x8 = torch.cat([x2, x3], dim=3)
        x8 = F.interpolate(x8, size=(x8.size(2), x1.size(3)), mode="bilinear", align_corners=True)
        x9 = torch.cat([x4, x6], dim=3)
        x9 = F.interpolate(x9, size=(x9.size(2), x1.size(3)), mode="bilinear", align_corners=True)
        x10 = torch.cat([x8, x9, x7], dim=2)
        x10 = F.interpolate(x10, size=(x1.size(2), x1.size(3)), mode="bilinear", align_corners=True)
        x10 = self.local_enhancement(x10)
        if self.use_gapw:
            x1, x10 = self.gapw(x1, x10)
        if self.use_bidirectional_interaction:
            x1, x10 = self.bidirectional_interaction(x1, x10)

        local_feat2 = torch.flatten(self.avgpool(x2), 1)
        local_feat3 = torch.flatten(self.avgpool(x3), 1)
        local_feat4 = torch.flatten(self.avgpool(x4), 1)
        local_feat6 = torch.flatten(self.avgpool(x6), 1)
        local_feat7 = torch.flatten(self.avgpool(x7), 1)
        head2 = self.fc2(local_feat2)
        head3 = self.fc3(local_feat3)
        head4 = self.fc4(local_feat4)
        head6 = self.fc6(local_feat6)
        head7 = self.fc7(local_feat7)
        heads = head2 + head3 + head4 + head6 + head7
        local_summary = (
            local_feat2 + local_feat3 + local_feat4 + local_feat6 + local_feat7
        ) / 5.0

        xs = self.features8(x10)
        xs = xs.view(x1.size(0), -1)
        if self.use_local_enhancement:
            delta_theta = torch.tanh(self.fc_loc(xs)).view(-1, 2, 3) * self.stn_delta_scale
            theta = self.identity_theta + delta_theta
        else:
            theta = self.fc_loc(xs).view(-1, 2, 3)
        grid = F.affine_grid(theta, x1.size(), align_corners=True)
        x11 = F.grid_sample(x10, grid, align_corners=True)

        x1 = x1 + x11
        x1 = self.features8(x1)
        x1 = torch.flatten(self.avgpool(x1), 1)
        out = self.fc(x1)
        if self.use_gated_fusion:
            out = self.gated_fusion(x1, local_summary, out, heads)
        return out, heads


class LocalEnhancementBlock(nn.Module):
    """Lightweight data-driven local feature enhancement.

    The block enriches the stitched local feature map with multi-scale context
    and uses a spatial attention mask to emphasize discriminative expression
    regions before PCNN's spatial transformer fusion.
    """

    def __init__(self, channels):
        super().__init__()
        hidden = channels // 4
        self.reduce = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False)
        self.branch5 = nn.Conv2d(hidden, hidden, kernel_size=5, padding=2, bias=False)
        self.fuse = nn.Sequential(
            nn.BatchNorm2d(hidden * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, channels // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        reduced = self.reduce(x)
        multi_scale = torch.cat([self.branch3(reduced), self.branch5(reduced)], dim=1)
        enhanced = self.fuse(multi_scale)
        attention = self.spatial_attention(enhanced)
        return x + self.gamma * enhanced * attention


class GlobalGuidedPatchWeighting(nn.Module):
    """Global-guided adaptive patch weighting for local PCNN features.

    The module turns PCNN's stitched local feature map into a 4x4 patch set.
    A global face context feature and each patch descriptor jointly predict
    patch reliability weights. The weighted local feature map is then used by
    the original PCNN STN fusion path. Residual scales are initialized to zero,
    so the module starts from the original PCNN behavior.
    """

    def __init__(self, channels, patch_grid=4, reduction=4, temperature=1.0, scale_limit=0.2):
        super().__init__()
        self.patch_grid = patch_grid
        self.temperature = max(float(temperature), 1e-3)
        self.scale_limit = float(scale_limit)
        hidden = max(channels // reduction, 32)
        self.patch_enhance = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.patch_score = nn.Sequential(
            nn.LayerNorm(channels * 3),
            nn.Linear(channels * 3, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        self.patch_reliability = nn.Sequential(
            nn.LayerNorm(channels * 3),
            nn.Linear(channels * 3, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )
        self.local_to_global = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )
        self.enhance_scale = nn.Parameter(torch.zeros(1))
        self.patch_scale = nn.Parameter(torch.zeros(1))
        self.global_scale = nn.Parameter(torch.zeros(1))
        self.last_patch_weights = None
        self.last_patch_reliability = None

    def forward(self, global_feat, local_feat):
        enhance_scale = torch.tanh(self.enhance_scale) * self.scale_limit
        patch_scale = torch.tanh(self.patch_scale) * self.scale_limit
        global_scale = torch.tanh(self.global_scale) * self.scale_limit

        enhanced_local = local_feat + enhance_scale * self.patch_enhance(local_feat)

        global_context = F.adaptive_avg_pool2d(global_feat, 1).flatten(1)
        patch_desc = F.adaptive_avg_pool2d(
            enhanced_local, (self.patch_grid, self.patch_grid)
        ).flatten(2).transpose(1, 2)
        global_context = global_context.unsqueeze(1).expand(-1, patch_desc.size(1), -1)
        patch_context = torch.cat(
            [global_context, patch_desc, torch.abs(global_context - patch_desc)],
            dim=-1,
        )

        patch_logits = self.patch_score(patch_context).squeeze(-1)
        patch_reliability = self.patch_reliability(patch_context).squeeze(-1)
        patch_logits = patch_logits + torch.log(patch_reliability.clamp_min(1e-4))
        patch_weights = torch.softmax(patch_logits / self.temperature, dim=1)
        self.last_patch_weights = patch_weights.detach()
        self.last_patch_reliability = patch_reliability.detach()

        weight_map = patch_weights.view(
            -1, 1, self.patch_grid, self.patch_grid
        )
        weight_map = F.interpolate(
            weight_map,
            size=enhanced_local.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
        normalized_weight_map = weight_map * (self.patch_grid * self.patch_grid)
        patch_residual = (normalized_weight_map - 1.0) * enhanced_local
        weighted_local = local_feat + patch_scale * patch_residual

        local_context = F.adaptive_avg_pool2d(weighted_local, 1).flatten(1)
        channel_gate = self.local_to_global(local_context).view(local_feat.size(0), -1, 1, 1)
        guided_global = global_feat * (1.0 + global_scale * channel_gate)
        return guided_global, weighted_local


class BidirectionalInteractionBlock(nn.Module):
    """Bidirectional guidance between global and local PCNN features.

    Global-to-local guidance produces a spatial mask from the whole-face feature
    map, helping stitched local features focus on expression-relevant regions.
    Local-to-global feedback produces channel weights from the guided local
    feature map, reinforcing global semantic channels supported by local cues.
    Zero-initialized residual scales make the block start as the original PCNN.
    """

    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 16)
        self.global_to_local = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self.local_to_global = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.local_scale = nn.Parameter(torch.zeros(1))
        self.global_scale = nn.Parameter(torch.zeros(1))

    def forward(self, global_feat, local_feat):
        spatial_gate = self.global_to_local(global_feat)
        guided_local = local_feat * (1.0 + self.local_scale * spatial_gate)

        channel_gate = self.local_to_global(guided_local)
        guided_global = global_feat * (1.0 + self.global_scale * channel_gate)
        return guided_global, guided_local


class GatedFusionHead(nn.Module):
    """Instance-aware fusion of PCNN global and local logits.

    The head learns a per-sample reliability gate from global fused features and
    summarized local-region features. It preserves PCNN's original main logits
    and adds a learned amount of the local auxiliary logits.
    """

    def __init__(self, global_dim, local_dim, num_class, hidden=128):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(global_dim + local_dim + num_class * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )
        self.gate_scale = nn.Parameter(torch.tensor(0.4))

    def forward(self, global_feat, local_feat, out, heads):
        gate_input = torch.cat([global_feat, local_feat, out.detach(), heads.detach()], dim=1)
        gate = self.gate(gate_input) * self.gate_scale
        return out + gate * heads
