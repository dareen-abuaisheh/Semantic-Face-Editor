from pathlib import Path

import matplotlib.pyplot as plt
import torch


def denorm(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(-1, 1) + 1) / 2


@torch.no_grad()
def save_triplet_grid(
    real: torch.Tensor,
    fake: torch.Tensor,
    reconstructed: torch.Tensor,
    out_path: str,
    max_items: int = 6,
) -> None:
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    real = denorm(real).cpu()
    fake = denorm(fake).cpu()
    reconstructed = denorm(reconstructed).cpu()

    n = min(max_items, real.size(0))
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = [axes]

    for i in range(n):
        axes[i][0].imshow(real[i].permute(1, 2, 0))
        axes[i][0].set_title("Original")
        axes[i][1].imshow(fake[i].permute(1, 2, 0))
        axes[i][1].set_title("Edited")
        axes[i][2].imshow(reconstructed[i].permute(1, 2, 0))
        axes[i][2].set_title("Reconstructed")
        for j in range(3):
            axes[i][j].axis("off")

    plt.tight_layout()
    plt.savefig(out_file, dpi=120)
    plt.close(fig)

