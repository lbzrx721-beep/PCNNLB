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

        self.fc = nn.Linear(512, num_class)
        self.fc2 = nn.Linear(256, num_class)
        self.fc3 = nn.Linear(256, num_class)
        self.fc4 = nn.Linear(256, num_class)
        self.fc6 = nn.Linear(256, num_class)
        self.fc7 = nn.Linear(256, num_class)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

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

        head2 = self.fc2(torch.flatten(self.avgpool(x2), 1))
        head3 = self.fc3(torch.flatten(self.avgpool(x3), 1))
        head4 = self.fc4(torch.flatten(self.avgpool(x4), 1))
        head6 = self.fc6(torch.flatten(self.avgpool(x6), 1))
        head7 = self.fc7(torch.flatten(self.avgpool(x7), 1))
        heads = head2 + head3 + head4 + head6 + head7

        xs = self.features8(x10)
        xs = xs.view(x1.size(0), -1)
        theta = self.fc_loc(xs).view(-1, 2, 3)
        grid = F.affine_grid(theta, x1.size(), align_corners=True)
        x11 = F.grid_sample(x10, grid, align_corners=True)

        x1 = x1 + x11
        x1 = self.features8(x1)
        x1 = torch.flatten(self.avgpool(x1), 1)
        out = self.fc(x1)
        return out, heads
