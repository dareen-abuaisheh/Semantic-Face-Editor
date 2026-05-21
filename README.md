# Interactive CelebA Attribute Filter Prototype (Toy, One-Day, AttGAN-Style)

## Project Structure

```text
project/
+-- data/
|   +-- img_align_celeba/
|   +-- list_attr_celeba.txt
+-- models/
|   +-- generator.py
|   +-- discriminator.py
+-- training/
|   +-- losses.py
|   +-- save_samples.py
|   +-- train.py
+-- demo/
|   +-- inference.py
|   +-- webcam_switch.py
+-- outputs/
|   +-- checkpoints/
|   +-- samples/
+-- dataset.py
```

## Important Note (Model Path X)

`PATH X = outputs/checkpoints/generator_latest.pt`

If this file is missing, download a compatible CelebA AttGAN-style checkpoint from online and place it exactly at:

`outputs/checkpoints/generator_latest.pt`

Quick search page:
`https://huggingface.co/models?search=attgan%20celeba`

## Minimal Install

```bash
pip install torch torchvision pillow matplotlib opencv-python mediapipe
```

## Multi-Attribute Setup (Default)

- Attributes: `Eyeglasses`, `Smiling`, `Male`
- Image size: `256`
- Batch size: `64`
- Device: `cuda` (GPU required for training)

## Pipeline (AttGAN-Style)

1. Encoder-decoder generator with attribute injection at bottleneck and decoder stages.
2. Discriminator with:
   - adversarial real/fake head
   - multi-attribute classification head
3. Unified loop losses:
   - adversarial loss
   - attribute classification loss
   - reconstruction loss
4. Random target attributes sampled per batch by flipping a subset of attributes.

## Inference / Demo

Run inference with an existing checkpoint:

```bash
python demo/inference.py --image path/to/face.jpg --attrs Eyeglasses Smiling Male --target-values 1 0 1
```

Optional webcam switching demo:

```bash
python demo/webcam_switch.py
```

## Training Command

```bash
python training/train.py --attrs Eyeglasses Smiling Male --image-size 256 --batch-size 64 --device cuda
```
