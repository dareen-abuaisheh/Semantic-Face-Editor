from pathlib import Path
import sys
import time

import cv2
import torch
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.generator import Generator  # noqa: E402

try:
    import mediapipe as mp
except ImportError as exc:
    raise ImportError("Please install mediapipe: pip install mediapipe") from exc


PATH_X = Path("outputs/checkpoints/generator_latest.pt")
ONLINE_NOTE = "https://huggingface.co/models?search=attgan%20celeba"
ATTRS = ["Eyeglasses", "Smiling", "Male"]
PRESETS = [
    ("Original", None),
    ("Eyeglasses", [1.0, 0.0, 0.0]),
    ("Smiling", [0.0, 1.0, 0.0]),
    ("Eyeglasses+Smiling", [1.0, 1.0, 0.0]),
    ("Eyeglasses+Smiling+Male", [1.0, 1.0, 1.0]),
]


def denorm_np(t: torch.Tensor):
    x = (t.clamp(-1, 1) + 1) / 2
    x = (x[0].permute(1, 2, 0).cpu().numpy() * 255.0).astype("uint8")
    return x


def preprocess_frame(frame_bgr, size=256):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tfm = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )
    return tfm(frame_rgb).unsqueeze(0)


def main():
    if not PATH_X.exists():
        raise FileNotFoundError(
            "Checkpoint missing.\n"
            f"IMPORTANT: download a compatible checkpoint online and put it at PATH X:\n{PATH_X}\n"
            f"Online search note: {ONLINE_NOTE}"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    G = Generator(num_attrs=len(ATTRS)).to(device)
    state = torch.load(PATH_X, map_location=device)
    if isinstance(state, dict) and "generator" in state:
        state = state["generator"]
    G.load_state_dict(state, strict=False)
    G.eval()

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    filter_idx = 0
    prev_x = None
    last_switch_time = 0.0
    cooldown_sec = 0.5
    swipe_threshold = 0.08

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        if result.multi_hand_landmarks:
            wrist = result.multi_hand_landmarks[0].landmark[0]
            x = wrist.x
            if prev_x is not None:
                dx = x - prev_x
                now = time.time()
                if now - last_switch_time > cooldown_sec:
                    if dx > swipe_threshold:
                        filter_idx = (filter_idx + 1) % len(PRESETS)
                        last_switch_time = now
                    elif dx < -swipe_threshold:
                        filter_idx = (filter_idx - 1) % len(PRESETS)
                        last_switch_time = now
            prev_x = x

        display = frame
        preset_name, preset_values = PRESETS[filter_idx]
        if preset_values is not None:
            x = preprocess_frame(frame).to(device)
            target = torch.tensor([preset_values], device=device, dtype=torch.float32)
            with torch.no_grad():
                y = G(x, target)
            edited_rgb = denorm_np(y)
            display = cv2.cvtColor(edited_rgb, cv2.COLOR_RGB2BGR)
            display = cv2.resize(display, (frame.shape[1], frame.shape[0]))

        label = f"Filter: {preset_name}"
        cv2.putText(display, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display, f"Attrs: {', '.join(ATTRS)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display, "Swipe right/left to switch. Press q to quit.", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Toy CelebA Webcam Switch", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
