from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CelebAAttributeDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        attr_path: str,
        selected_attrs: List[str],
        image_size: int = 64,
        max_images: int = 5000,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.attr_path = Path(attr_path)
        self.selected_attrs = selected_attrs
        self.image_size = image_size
        self.max_images = max_images

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        self.samples = self._load_samples()

    def _load_samples(self) -> List[Tuple[str, torch.Tensor]]:
        lines = self.attr_path.read_text(encoding="utf-8").splitlines()
        attr_names = lines[1].split()
        attr_to_idx = {name: idx for idx, name in enumerate(attr_names)}

        selected_indices = []
        for attr in self.selected_attrs:
            if attr not in attr_to_idx:
                raise ValueError(f"Attribute '{attr}' not found in list_attr_celeba.txt")
            selected_indices.append(attr_to_idx[attr])

        samples = []
        for line in lines[2 : 2 + self.max_images]:
            parts = line.split()
            filename = parts[0]
            raw_values = parts[1:]

            attrs = []
            for idx in selected_indices:
                # CelebA format: 1 means present, -1 means absent.
                attrs.append(1.0 if raw_values[idx] == "1" else 0.0)

            samples.append((filename, torch.tensor(attrs, dtype=torch.float32)))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        filename, attrs = self.samples[idx]
        image_path = self.image_dir / filename
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, attrs


def sample_target_attributes(
    attrs: torch.Tensor,
    flip_prob: float = 0.5,
    ensure_change: bool = True,
) -> torch.Tensor:
    """
    Build AttGAN-style target attributes by randomly flipping a subset of attrs.
    attrs shape: [B, C]
    """
    if attrs.dim() != 2:
        raise ValueError(f"Expected attrs shape [B, C], got {tuple(attrs.shape)}")

    flip_mask = (torch.rand_like(attrs) < flip_prob).float()
    target = attrs * (1.0 - flip_mask) + (1.0 - attrs) * flip_mask

    if ensure_change:
        changed = (target != attrs).any(dim=1)
        if not changed.all():
            for i in torch.where(~changed)[0]:
                j = torch.randint(0, attrs.size(1), (1,), device=attrs.device).item()
                target[i, j] = 1.0 - target[i, j]
    return target
