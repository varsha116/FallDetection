#!/usr/bin/env python3
"""Run fall detection on a video using YOLOv8 and CLIP."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from ultralytics import YOLO

CLIP_TEXT_PROMPTS = ["a person falling", "a person standing"]


@dataclass
class ModelBundle:
    """Container for shared YOLO + CLIP models."""

    device: torch.device
    yolo: YOLO
    clip_model: CLIPModel
    clip_processor: CLIPProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect falls in a video by combining YOLOv8 detections with CLIP classification.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path where the annotated video will be saved. Defaults to <input>_fall_detected.mp4",
    )
    parser.add_argument(
        "--yolo-model",
        default="yolov8n.pt",
        help="YOLOv8 weights to use for person detection.",
    )
    parser.add_argument(
        "--clip-model",
        default="openai/clip-vit-base-patch32",
        help="Hugging Face model id for CLIP.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Torch device for CLIP. 'auto' picks the best available option.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Minimum YOLO confidence to keep a detection.",
    )
    parser.add_argument(
        "--fall-threshold",
        type=float,
        default=0.5,
        help="Probability threshold from CLIP to classify a detection as a fall.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=60,
        help="How often (in frames) to print progress updates.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if device_arg == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.", file=sys.stderr)
        return torch.device("cpu")

    if device_arg == "mps" and not torch.backends.mps.is_available():
        print("MPS requested but not available. Falling back to CPU.", file=sys.stderr)
        return torch.device("cpu")

    return torch.device(device_arg)


def clip_fall_probability(
    clip_processor: CLIPProcessor,
    clip_model: CLIPModel,
    crop_rgb,
    device: torch.device,
) -> float:
    """Return the probability that the cropped person is falling."""
    pil_image = Image.fromarray(crop_rgb)
    inputs = clip_processor(
        text=CLIP_TEXT_PROMPTS,
        images=pil_image,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        outputs = clip_model(**inputs)
        logits = outputs.logits_per_image  # (1, len(CLIP_TEXT_PROMPTS))
        probs = logits.softmax(dim=-1)
    return float(probs[0][0])


def create_writer(output_path: Path, frame_size: Tuple[int, int], fps: float) -> cv2.VideoWriter:
    width, height = frame_size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}.")
    return writer


def load_model_bundle(yolo_model: str, clip_model_id: str, device_choice: str) -> ModelBundle:
    """Load YOLO + CLIP once for reuse in CLIs or web apps."""
    device = resolve_device(device_choice)
    print(f"Using CLIP on device: {device}")

    print(f"Loading YOLO model '{yolo_model}'...")
    yolo = YOLO(yolo_model)

    print(f"Loading CLIP model '{clip_model_id}'...")
    clip_model = CLIPModel.from_pretrained(clip_model_id)
    clip_model.to(device)
    clip_model.eval()
    clip_processor = CLIPProcessor.from_pretrained(clip_model_id)

    return ModelBundle(device=device, yolo=yolo, clip_model=clip_model, clip_processor=clip_processor)


def run_fall_detection(
    input_path: Path | str,
    output_path: Path | str,
    *,
    confidence: float,
    fall_threshold: float,
    log_every: int,
    models: ModelBundle,
) -> Path:
    """Annotate the input video and write the result to output_path."""
    input_path = Path(input_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0  # fallback if fps metadata is missing

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    frame_idx = 0
    writer = None
    frame_size: Optional[Tuple[int, int]] = None
    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            if writer is None:
                frame_size = (frame.shape[1], frame.shape[0])
                writer = create_writer(output_path, frame_size, fps)
                if total_frames and frame_size:
                    print(
                        f"Processing video with {total_frames} frames "
                        f"at {fps:.2f} FPS ({frame_size[0]}x{frame_size[1]})."
                    )
                else:
                    print(f"Processing video at {fps:.2f} FPS ({frame_size[0]}x{frame_size[1]}).")

            results = models.yolo(frame, verbose=False)
            if results:
                result = results[0]
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        cls_id = int(box.cls[0].item())
                        if cls_id != 0:  # 0 == person in COCO
                            continue

                        confidence_score = float(box.conf[0].item())
                        if confidence_score < confidence:
                            continue

                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(frame.shape[1], x2)
                        y2 = min(frame.shape[0], y2)
                        if x2 <= x1 or y2 <= y1:
                            continue

                        person_crop = frame[y1:y2, x1:x2]
                        person_crop_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)

                        fall_prob = clip_fall_probability(
                            models.clip_processor, models.clip_model, person_crop_rgb, models.device
                        )
                        is_fall = fall_prob >= fall_threshold

                        color = (0, 0, 255) if is_fall else (0, 200, 0)
                        label = f"{'FALL' if is_fall else 'Safe'} {fall_prob:.2f}"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(
                            frame,
                            label,
                            (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2,
                            cv2.LINE_AA,
                        )

            if writer is None:
                raise RuntimeError("Video writer was not initialized.")
            writer.write(frame)

            frame_idx += 1
            if log_every and frame_idx % log_every == 0:
                print(f"Processed {frame_idx} frames...")

        print(f"Fall detection complete. Output saved to: {output_path}")
        return output_path
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()

    input_path = args.input.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else input_path.with_name(f"{input_path.stem}_fall_detected.mp4")
    )

    models = load_model_bundle(args.yolo_model, args.clip_model, args.device)
    run_fall_detection(
        input_path=input_path,
        output_path=output_path,
        confidence=args.confidence,
        fall_threshold=args.fall_threshold,
        log_every=args.log_every,
        models=models,
    )


if __name__ == "__main__":
    main()

