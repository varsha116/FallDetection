"""Gradio app that lets users upload a video and download the fall-detection result."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from uuid import uuid4

import gradio as gr

from src.fall_detection import load_model_bundle, run_fall_detection


def _default_yolo_model() -> str:
    local_weight = Path("weights/yolov8n.pt")
    return str(local_weight) if local_weight.exists() else "yolov8n.pt"


YOLO_MODEL = os.environ.get("FALL_YOLO_MODEL", _default_yolo_model())
CLIP_MODEL = os.environ.get("FALL_CLIP_MODEL", "openai/clip-vit-base-patch32")
DEVICE = os.environ.get("FALL_DEVICE", "cpu")
CONFIDENCE = float(os.environ.get("FALL_CONFIDENCE", "0.5"))
FALL_THRESHOLD = float(os.environ.get("FALL_THRESHOLD", "0.5"))

MODEL_BUNDLE = load_model_bundle(YOLO_MODEL, CLIP_MODEL, DEVICE)
TMP_DIR = Path(tempfile.gettempdir())


def process_video(video_path: str | None) -> str:
    """Gradio callback."""
    if not video_path:
        raise gr.Error("Please upload a short MP4 or MOV clip.")

    src_path = Path(video_path)
    suffix = src_path.suffix or ".mp4"
    session_id = uuid4().hex
    input_path = TMP_DIR / f"fall_input_{session_id}{suffix}"
    output_path = TMP_DIR / f"fall_output_{session_id}.mp4"

    shutil.copy(src_path, input_path)
    run_fall_detection(
        input_path=input_path,
        output_path=output_path,
        confidence=CONFIDENCE,
        fall_threshold=FALL_THRESHOLD,
        log_every=0,
        models=MODEL_BUNDLE,
    )
    return str(output_path)


DESCRIPTION = """
Upload a short video (<= ~20s), and the app will draw red boxes for falls and green boxes for
everyone else using YOLOv8 detections + CLIP classification. CPU runtimes can take a minute; GPU Spaces
run much faster.
"""

demo = gr.Interface(
    fn=process_video,
    inputs=gr.Video(label="Input video", format="mp4"),
    outputs=gr.Video(label="Annotated output"),
    title="Fall Detection (YOLOv8 + CLIP)",
    description=DESCRIPTION,
    allow_flagging="never",
)


if __name__ == "__main__":
    demo.launch()

