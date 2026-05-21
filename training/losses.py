import torch
import torch.nn.functional as F


def d_adversarial_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    real_targets = torch.ones_like(real_logits)
    fake_targets = torch.zeros_like(fake_logits)
    real_loss = F.binary_cross_entropy_with_logits(real_logits, real_targets)
    fake_loss = F.binary_cross_entropy_with_logits(fake_logits, fake_targets)
    return real_loss + fake_loss


def g_adversarial_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    targets = torch.ones_like(fake_logits)
    return F.binary_cross_entropy_with_logits(fake_logits, targets)


def attribute_loss(attr_logits: torch.Tensor, attrs: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(attr_logits, attrs)


def reconstruction_loss(reconstructed: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(reconstructed, real)

