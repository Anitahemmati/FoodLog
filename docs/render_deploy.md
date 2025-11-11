# Render Deployment Guide

This document describes the quickest way to run the Food Log backend without managing AWS directly. Render provides a managed container service with persistent disks that works well with the project’s Dockerfile and SQLite backend.

## 1. Prerequisites
- Render account (https://render.com).
- Git repository that contains this backend (Push your current branch to GitHub/GitLab/Bitbucket).
- Docker is **not** required on Render, but you can build locally with `docker build -t foodlog-backend:latest .` if you want to test first.

## 2. Review configuration
Ensure the repository includes:
- `Dockerfile` (already present).
- `render.yaml` (added in the project root by this guide).
- `.env` or secrets are **not** committed. Render will prompt you for sensitive values such as `JWT_SECRET_KEY`.

Environment variables Render uses (from `render.yaml`):
- `STORAGE_BACKEND=sqlite` – runs entirely on the container disk.
- `UPLOADS_DIR=/data/uploads` and `SQLITE_DB_PATH=/data/db/food_history.db` – both stored on the persistent disk so data survives restarts.
- `ML_SERVICE_URL=stub` while the real ML service is unavailable.
- `ML_SERVICE_API_KEY` (leave empty unless needed).
- `JWT_SECRET_KEY` – set this in the Render dashboard (use `openssl rand -hex 32` locally to generate a strong value).
- `HF_API_TOKEN` – required. Paste your Hugging Face token (e.g., `hf_zLQ...`) in Render → **Environment** so the backend can call the classifier.
- `HF_SPACE_URL=https://huggingface.co/spaces/AnitaAnita2025/nateraw-food/+/api/predict/` – adjust if you point to a different space or switch back to the nateraw endpoint.
- `HF_CONFIDENCE_THRESHOLD=0.5` – tweak if you want to require higher or lower confidence before accepting a prediction (expressed as a ratio, so 0.5 = 50 %).
- `HF_SPACE_TIMEOUT=45` – network timeout (seconds) for the Hugging Face request.
- TensorFlow and the training utilities are optional; runtime predictions now use synthetic data so the container no longer ships heavy ML dependencies. If you need to retrain models locally, install `tensorflow` manually before running `train_model.py`.

## 3. Deploy via Render Blueprint
1. Push your latest changes to the default branch.
2. In Render dashboard select **New +** → **Blueprint**.
3. Paste the repository URL and choose the branch that contains `render.yaml`.
4. Render will parse the blueprint and present a `foodlog-backend` web service.
5. Hit **Create Resources**. The first build takes a few minutes because Render builds the Docker image.

The blueprint attaches a 1 GB persistent disk at `/data`, so uploaded images and the SQLite database are retained between deploys.

## 4. Validate the deployment
After the service reports “Live”:
```bash
curl https://<your-render-subdomain>.onrender.com/health
```
You should receive `{"status":"ok", ...}`.

To sign up a user:
```bash
curl -X POST https://<your-render-subdomain>.onrender.com/auth/signup \
  -H "Content-Type: application/json" \
  -d '{"email":"demo@foodlog.app","password":"Str0ngPass!"}'
```

For predictions, call `/predict` with `multipart/form-data`. The response uses the stubbed ML output until a real service is configured.

## 5. Frontend configuration
Update the frontend API base URL to the Render hostname (e.g., `https://foodlog-backend.onrender.com`). Once committed and pushed, redeploy the frontend in Vercel so both services point at the same backend.

## 6. Next steps
- When the ML microservice is ready, update `ML_SERVICE_URL` (and optionally `ML_SERVICE_API_KEY`) in Render → **Environment** and trigger a redeploy.
- If the Hugging Face Space token rotates, edit `HF_API_TOKEN` in Render and hit **Manual Deploy → Deploy latest commit** to roll out the new secret.
- If you later migrate to AWS or another provider, remove `render.yaml` or keep both deployment options side by side.
