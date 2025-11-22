# Fall Detection (YOLOv8 + CLIP)

This repo pairs **YOLOv8** person detections with **OpenAI CLIP** to decide whether each detected person is falling. The original proof-of-concept lived in a Colab notebook; the project now includes a standalone Python script you can run locally (or inside VS Code) on any MP4 video.

```
├── README.md
├── requirements.txt
├── src/
│   └── fall_detection.py
├── notebooks/
│   └── FallDetection_CollaborationWithClipModel.ipynb
├── samples/                # Optional demo/input videos
├── weights/                # Optional local YOLO weights cache
└── docs/                  # Supplementary material (e.g., Colab link)
```

## Features
- YOLOv8 person detector (default `yolov8n.pt`) for tight bounding boxes.
- CLIP text prompts (`"a person falling"` vs `"a person standing"`) to classify each detection.
- Command-line tool `src/fall_detection.py` that accepts any input video and writes an annotated MP4.
- Optional Gradio web UI (`app.py`) ready for Hugging Face Spaces / Streamlit / local preview.
- Works on CPU, CUDA, or Apple MPS (auto-selection by default).

## How it works (beginner friendly)
1. **Detect persons:** YOLOv8 looks at each frame and draws bounding boxes for people.
2. **Crop regions:** We crop each person box so CLIP only sees that person.
3. **Compare prompts:** CLIP scores the crop against `"a person falling"` vs `"a person standing"`.
4. **Threshold:** If the “falling” probability ≥ threshold (default 0.5) we color the box red, otherwise green.
5. **Write video:** Annotated frames are saved back to a video so you can replay the results.

Because CLIP is language-driven, you can experiment with prompts (e.g., “a person lying on the floor”) without training a new model.

## Requirements
- Python 3.10+ (tested on 3.11).
- A recent `pip`. For GPU acceleration, install the matching **PyTorch** build from [pytorch.org](https://pytorch.org/get-started/locally/) before `pip install -r requirements.txt`.
- Windows 10/11, macOS, or Linux.

## Setup (VS Code friendly)
1. **Open the folder** `FallDetection-main` in VS Code.
2. **Create a virtual environment** (PowerShell example):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   ```
3. **Install dependencies** (PowerShell prefers `python -m pip`):
   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```
   > If PyTorch GPU wheels fail to install on Windows, install the appropriate wheel from pytorch.org first, then rerun the command above (it will reuse the existing torch install).
4. **Configure VS Code** (optional but recommended):
   - Select the `.venv` interpreter (`Ctrl+Shift+P` → *Python: Select Interpreter*).
   - Install the Python extension if prompted.

## Usage
Run the detector from the VS Code terminal (or any shell):

```powershell
python src/fall_detection.py --input path\to\video.mp4 --output path\to\annotated.mp4
```

Key arguments:

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | **required** | Source video (MP4 recommended). |
| `--output` | `<input>_fall_detected.mp4` | Destination file. |
| `--yolo-model` | `yolov8n.pt` | YOLO weights; replace with `yolov8s.pt`, etc. |
| `--clip-model` | `openai/clip-vit-base-patch32` | Hugging Face CLIP checkpoint. |
| `--device` | `auto` | Chooses CUDA/MPS/CPU. |
| `--confidence` | `0.5` | Minimum YOLO confidence for a detection. |
| `--fall-threshold` | `0.5` | CLIP probability to flag a fall. |

Example (CPU only, custom thresholds):

```powershell
python src/fall_detection.py --input samples\insta_FALL.mp4 --device cpu --confidence 0.4 --fall-threshold 0.6
```

Outputs are written as standard MP4 videos with red boxes for falls and green boxes for non-falls.

## Web demo (Gradio / Hugging Face Spaces)
`app.py` spins up a Gradio UI where users upload a short video and download the annotated result:

```powershell
python -m pip install -r requirements.txt  # gradio is included
python app.py  # launches on http://127.0.0.1:7860
```

Deploy for free:
1. Create a [Hugging Face](https://huggingface.co) account.
2. Click **New Space → Gradio → Blank** and select the free CPU option.
3. Point the Space to this GitHub repo (or upload the files manually). Spaces install `requirements.txt`, run `app.py`, and give you a public URL.

Detailed Git + Spaces steps live in `docs/DEPLOYMENT.md`.

## Notes & Troubleshooting
- **Performance:** CLIP runs per detected person, so CPU mode can be slow. Prefer GPU for real-time-ish throughput.
- **Custom prompts:** Edit `CLIP_TEXT_PROMPTS` inside `src/fall_detection.py` to experiment with different descriptions.
- **No frames read:** Ensure the video codec is supported by OpenCV (`opencv-video` backend). Converting to MP4/H264 usually helps.
- **YOLO weights:** The script downloads YOLOv8 weights automatically; a local copy can live in `weights/` if you want to avoid repeat downloads.
- **Existing notebook:** `notebooks/FallDetection_CollaborationWithClipModel.ipynb` remains for reference but is no longer required to run the detector.

## GitHub + Free hosting quickstart
- `git init`, `git add .`, `git commit -m "Initial commit"`, then `git remote add origin <your GitHub URL>` and `git push -u origin main`.
- Reuse the same repo as the source for Hugging Face Spaces so every push redeploys your free web demo automatically.
- See `docs/DEPLOYMENT.md` for copy/paste commands (plus a Streamlit Cloud alternative).
