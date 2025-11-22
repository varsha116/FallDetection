## Deploying the Fall Detection Project

This guide shows how to (a) publish the repo on GitHub and (b) expose a free web link where anyone can upload a video and download the annotated result. No paid services are required.

---

## 1. Push the project to GitHub
1. Create a new empty repository on GitHub (no README/license).
2. In VS Code’s terminal (inside the project folder):
   ```powershell
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/<username>/<repo>.git
   git push -u origin main
   ```
3. Future updates:
   ```powershell
   git add .
   git commit -m "Describe your change"
   git push
   ```

---

## 2. Free web link via Hugging Face Spaces (Gradio)
1. Create a free Hugging Face account and verify your email.
2. Click **New Space**:
   - **SDK:** Gradio
   - **Space name:** e.g., `username/fall-detection-demo`
   - **Hardware:** CPU basic (free)
3. Choose **“Create from Git Repository”** and paste your GitHub URL, or upload the files manually.
4. Spaces auto-run:
   - `pip install -r requirements.txt`
   - `python app.py` (because `app.py` instantiates a Gradio interface)
5. After the build finishes, the top-right **“App”** button is your shareable URL. Each new `git push` to main re-triggers the build.

### Optional tweaks
- **Environment vars:** In the Space settings you can add secrets like `FALL_DEVICE=cuda` if you enable a paid GPU later.
- **Demo text:** Edit `README.md` to include the Space link so visitors can find it easily.

---

## 3. Alternative: Streamlit Cloud (also free)
1. Create a [Streamlit Cloud](https://streamlit.io/cloud) account.
2. Click **“New app”** and select your GitHub repo + branch.
3. Set the entry point to `app.py`. Streamlit installs `requirements.txt` and gives you a public link similar to Hugging Face Spaces.

> Streamlit expects Streamlit code, but Gradio apps can be wrapped via `gradio.streamlit.GradioProxy`. If you prefer native Streamlit, create `streamlit_app.py` that imports and uses `run_fall_detection` on upload.

---

## 4. Learning checklist
- **Model flow:** Understand how YOLO detects people and CLIP classifies crops (see `README.md` → “How it works”).
- **Code reading:** Open `src/fall_detection.py` to follow the processing loop end-to-end.
- **Web UI:** Inspect `app.py` to see how Gradio uploads map to filesystem paths and call the shared `run_fall_detection` logic.
- **Deployment:** Practice the Git commands above, then redeploy a small change to your Space to see CI/CD in action.

You now have a GitHub-hosted project plus a free, public URL to share results without running Colab or paying for servers. Happy hacking!

