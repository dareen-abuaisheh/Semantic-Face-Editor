from pathlib import Path
import argparse
import sys

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.generator import Generator  # noqa: E402


PATH_X = Path("outputs/checkpoints/generator_latest.pt")
ONLINE_NOTE = "https://huggingface.co/models?search=attgan%20celeba"


def parse_args():
    parser = argparse.ArgumentParser(description="Toy AttGAN-style multi-attribute inference")
    parser.add_argument("--image", type=str, required=True, help="Input face image path")
    parser.add_argument("--checkpoint", type=str, default=str(PATH_X))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--attrs", nargs="+", default=["Eyeglasses", "Smiling", "Male"])
    parser.add_argument(
        "--target-values",
        nargs="+",
        type=float,
        default=None,
        help="Binary targets for attrs, e.g. --target-values 1 0 1",
    )
    parser.add_argument("--out", type=str, default="outputs/demo_result.jpg")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_image(path: str, image_size: int) -> torch.Tensor:
    tfm = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    image = Image.open(path).convert("RGB")
    return tfm(image).unsqueeze(0)


def denorm(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(-1, 1) + 1) / 2


def main():
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            "Checkpoint missing.\n"
            f"IMPORTANT: download a compatible checkpoint online and put it at PATH X:\n{PATH_X}\n"
            f"Online search note: {ONLINE_NOTE}"
        )

    device = args.device
    num_attrs = len(args.attrs)
    G = Generator(num_attrs=num_attrs).to(device)
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "generator" in state:
        state = state["generator"]
    G.load_state_dict(state, strict=False)
    G.eval()

    if args.target_values is None:
        target_values = [1.0] * num_attrs
    else:
        if len(args.target_values) != num_attrs:
            raise ValueError(
                f"target-values length ({len(args.target_values)}) must match attrs length ({num_attrs})"
            )
        target_values = args.target_values

    image = load_image(args.image, args.image_size).to(device)
    target_attr = torch.tensor([target_values], device=device, dtype=torch.float32)

    with torch.no_grad():
        edited = G(image, target_attr)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save side-by-side: original | edited
    merged = torch.cat([denorm(image), denorm(edited)], dim=3)
    save_image(merged, str(out_path))
    print(f"Saved: {out_path}")
    print(f"Attrs: {args.attrs}")
    print(f"Target values: {target_values}")


if __name__ == "__main__":
    main()
