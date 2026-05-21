import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """
    Outputs:
      adv_logits: [B, 1, H', W'] (PatchGAN logits)
      attr_logits: [B, C_attr]
    """

    def __init__(self, image_size: int = 256, num_attrs: int = 1, base_channels: int = 64, repeat_num: int = 6) -> None:
        super().__init__()
        layers = []
        in_channels = 3
        out_channels = base_channels

        for i in range(repeat_num):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01, inplace=True))
            in_channels = out_channels
            out_channels = min(out_channels * 2, 512)

        self.features = nn.Sequential(*layers)
        self.adv_head = nn.Conv2d(in_channels, 1, kernel_size=3, stride=1, padding=1, bias=False)

        kernel_size = image_size // (2**repeat_num)
        self.attr_head = nn.Conv2d(in_channels, num_attrs, kernel_size=kernel_size, bias=False)

    def forward(self, x: torch.Tensor):
        feat = self.features(x)
        adv_logits = self.adv_head(feat)
        attr_logits = self.attr_head(feat).view(x.size(0), -1)
        return adv_logits, attr_logits
