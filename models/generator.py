import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(channels, affine=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class Generator(nn.Module):
    """
    Input:
      image: [B, 3, H, W]
      target_attrs: [B, C_attr]
    Output:
      fake: [B, 3, H, W]
    """

    def __init__(self, num_attrs: int = 1, base_channels: int = 64, num_res_blocks: int = 4) -> None:
        super().__init__()
        self.num_attrs = num_attrs

        # AttGAN-style: encode image first, inject attributes at bottleneck.
        self.down = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=7, stride=1, padding=3, bias=False),
            nn.InstanceNorm2d(base_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels * 2, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels * 4, affine=True),
            nn.ReLU(inplace=True),
        )

        res = []
        for _ in range(num_res_blocks):
            res.append(ResidualBlock(base_channels * 4))
        self.res = nn.Sequential(*res)

        self.bottleneck_fuse = nn.Sequential(
            nn.Conv2d(base_channels * 4 + num_attrs, base_channels * 4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(base_channels * 4, affine=True),
            nn.ReLU(inplace=True),
        )

        # Inject attributes again before each upsample stage to strengthen control.
        up_in_1 = base_channels * 4 + num_attrs
        up_in_2 = base_channels * 2 + num_attrs
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(up_in_1, base_channels * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels * 2, affine=True),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(up_in_2, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 3, kernel_size=7, stride=1, padding=3, bias=True),
            nn.Tanh(),
        )

    def forward(self, image: torch.Tensor, target_attrs: torch.Tensor) -> torch.Tensor:
        if target_attrs.dim() == 1:
            target_attrs = target_attrs.unsqueeze(1)

        b, _, _, _ = image.shape
        x = self.down(image)
        x = self.res(x)

        attr_map_low = target_attrs.view(b, -1, 1, 1)
        attr_map_low = F.interpolate(attr_map_low, size=(x.size(2), x.size(3)), mode="nearest")
        x = torch.cat([x, attr_map_low], dim=1)
        x = self.bottleneck_fuse(x)

        x = self.up[0](x)
        attr_map_mid = F.interpolate(attr_map_low, size=(x.size(2), x.size(3)), mode="nearest")
        x = torch.cat([x, attr_map_mid], dim=1)
        x = self.up[1](x)
        x = self.up[2](x)

        x = self.up[3](x)
        attr_map_high = F.interpolate(attr_map_low, size=(x.size(2), x.size(3)), mode="nearest")
        x = torch.cat([x, attr_map_high], dim=1)
        x = self.up[4](x)
        x = self.up[5](x)
        x = self.up[6](x)
        x = self.up[7](x)

        return x
