# annotate-web — iris/pupil annotation tool

Crowd-sourced web annotator for the iris/pupil segmentation dataset. The
frontend pre-segments a webcam frame with the trained model so the user only
has to correct what's wrong, and the backend stores the labels alongside the
original image.

## Pieces

```
backend/        FastAPI + slowapi (rate limit) + sqlite + on-disk image storage
  app.py          REST endpoints
  db.py           sqlite schema + migrations
  storage.py      image bytes on disk, IP hashing
  migrations/     forward-only SQL files
frontend/       Astro + React + onnxruntime-web (same stack as demo-web)
  src/components/steps/   the labelling wizard steps
  lib/                    crop/preprocess/onnx/render (forked from demo-web)
  public/models/          two grayscale ONNX models for in-browser pre-segmentation
```

## Quick start

```bash
docker compose up           # backend on :8000, frontend on :4322
# or, locally:
cd backend && pip install -r requirements.txt && uvicorn app:app --reload
cd ../frontend && npm install && npm run dev
```

## Env vars (backend)

- `DATABASE_PATH` — sqlite file (default `./annotate.db`)
- `STORAGE_DIR` — image storage root (default `./storage`)
- `CORS_ORIGINS` — comma-separated origins
- `IP_HASH_SALT` — salt for hashed IP storage

## Frontend env vars

- `PUBLIC_API_URL` — backend base URL (default `http://localhost:8000`)
