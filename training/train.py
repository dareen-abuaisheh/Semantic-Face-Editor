import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import CelebAAttributeDataset, sample_target_attributes
from models.generator import Generator
from models.discriminator import Discriminator
from training.losses import (
    attribute_loss,
    d_adversarial_loss,
    g_adversarial_loss,
    reconstruction_loss,
)
from training.save_samples import save_triplet_grid


def parse_args():
    parser = argparse.ArgumentParser(description="Toy AttGAN-style multi-attribute trainer (CelebA)")
    parser.add_argument("--image-dir", type=str, default="data/img_align_celeba")
    parser.add_argument("--attr-path", type=str, default="data/list_attr_celeba.txt")
    parser.add_argument("--attrs", nargs="+", default=["Eyeglasses", "Smiling", "Male"])
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--max-images", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--lambda-cls", type=float, default=1.0)
    parser.add_argument("--lambda-attr", type=float, default=10.0)
    parser.add_argument("--lambda-rec", type=float, default=10.0)
    parser.add_argument("--flip-prob", type=float, default=0.5)
    parser.add_argument("--sample-every", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU training is required, but CUDA is not available.")
    if not args.device.startswith("cuda"):
        raise ValueError("Training is configured for GPU. Use a CUDA device, e.g. --device cuda")

    out_ckpt_dir = Path("outputs/checkpoints")
    out_samples_dir = Path("outputs/samples")
    out_ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_samples_dir.mkdir(parents=True, exist_ok=True)

    dataset = CelebAAttributeDataset(
        image_dir=args.image_dir,
        attr_path=args.attr_path,
        selected_attrs=args.attrs,
        image_size=args.image_size,
        max_images=args.max_images,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)

    num_attrs = len(args.attrs)
    G = Generator(num_attrs=num_attrs).to(args.device)
    D = Discriminator(image_size=args.image_size, num_attrs=num_attrs).to(args.device)

    g_opt = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    d_opt = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    step = 0
    for epoch in range(args.epochs):
        for real_images, original_attrs in loader:
            real_images = real_images.to(args.device)
            original_attrs = original_attrs.to(args.device)
            target_attrs = sample_target_attributes(original_attrs, flip_prob=args.flip_prob)

            with torch.no_grad():
                fake_images = G(real_images, target_attrs)

            real_logits, real_attr_logits = D(real_images)
            fake_logits, _ = D(fake_images)

            d_adv = d_adversarial_loss(real_logits, fake_logits)
            d_cls = attribute_loss(real_attr_logits, original_attrs)
            d_loss = d_adv + args.lambda_cls * d_cls

            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()

            fake_images = G(real_images, target_attrs)
            fake_logits, fake_attr_logits = D(fake_images)
            reconstructed_images = G(fake_images, original_attrs)

            g_adv = g_adversarial_loss(fake_logits)
            g_cls = attribute_loss(fake_attr_logits, target_attrs)
            g_rec = reconstruction_loss(reconstructed_images, real_images)
            g_loss = g_adv + args.lambda_attr * g_cls + args.lambda_rec * g_rec

            g_opt.zero_grad()
            g_loss.backward()
            g_opt.step()

            if step % 50 == 0:
                print(
                    f"[Epoch {epoch+1}/{args.epochs}] [Step {step}] "
                    f"D: {d_loss.item():.4f} | G: {g_loss.item():.4f} | "
                    f"Rec: {g_rec.item():.4f} | Attrs: {','.join(args.attrs)}"
                )

            if step % args.sample_every == 0:
                sample_file = out_samples_dir / f"sample_step_{step:06d}.png"
                save_triplet_grid(real_images, fake_images, reconstructed_images, str(sample_file))

            if step % args.save_every == 0:
                torch.save(
                    {
                        "step": step,
                        "attrs": args.attrs,
                        "generator": G.state_dict(),
                        "discriminator": D.state_dict(),
                    },
                    out_ckpt_dir / f"ckpt_step_{step:06d}.pt",
                )
                torch.save(G.state_dict(), out_ckpt_dir / "generator_latest.pt")

            step += 1


if __name__ == "__main__":
    main()
