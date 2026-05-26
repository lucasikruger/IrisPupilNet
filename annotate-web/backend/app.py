import json
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

import db
import storage

# slowapi 0.1.9 calls key_func with no args internally; the documented pattern
# is to pass get_remote_address directly. Hashing for storage happens in
# storage.hash_ip at write-time, separately from the rate-limit key.
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    storage.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    await db.init_db()
    try:
        yield
    finally:
        await db.close_db()


app = FastAPI(lifespan=lifespan)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429, content={"ok": False, "error": "rate-limited"})


cors_origins = [
    o.strip()
    for o in os.environ.get("CORS_ORIGINS", "http://localhost:4322,http://localhost:4323").split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/healthz")
async def healthz():
    return {"ok": True}


@app.post("/api/submit")
@limiter.limit("5/hour")
async def submit(
    request: Request,
    full: UploadFile = File(...),
    crop_left: UploadFile = File(...),
    crop_right: UploadFile = File(...),
    metadata: UploadFile = File(...),
):
    try:
        md = json.loads((await metadata.read()).decode("utf-8"))
    except Exception:
        raise HTTPException(status_code=400, detail="invalid metadata json")

    sub_id, dir_path, rel = storage.new_submission_path()
    (dir_path / "full.jpg").write_bytes(await full.read())
    (dir_path / "crop_left.png").write_bytes(await crop_left.read())
    (dir_path / "crop_right.png").write_bytes(await crop_right.read())
    storage.write_metadata_json(dir_path, md)

    ip = (request.client.host if request.client else "0.0.0.0")
    await db.insert_submission(sub_id, storage.hash_ip(ip), md, rel)
    return {"ok": True, "id": sub_id}


@app.post("/api/submit/{sub_id}/refine")
@limiter.limit("10/hour")
async def refine(
    sub_id: str,
    request: Request,
    mask_iris_left: UploadFile = File(...),
    mask_pupil_left: UploadFile = File(...),
    mask_eyelid_left: UploadFile = File(...),
    mask_iris_right: UploadFile = File(...),
    mask_pupil_right: UploadFile = File(...),
    mask_eyelid_right: UploadFile = File(...),
    geometry: UploadFile = File(...),
):
    rel = await db.get_storage_path(sub_id)
    if rel is None:
        raise HTTPException(status_code=404, detail="submission not found")
    dir_path = storage.submission_dir(rel)

    uploads = {
        "mask_iris_left.png": mask_iris_left,
        "mask_pupil_left.png": mask_pupil_left,
        "mask_eyelid_left.png": mask_eyelid_left,
        "mask_iris_right.png": mask_iris_right,
        "mask_pupil_right.png": mask_pupil_right,
        "mask_eyelid_right.png": mask_eyelid_right,
    }
    for fname, upload in uploads.items():
        (dir_path / fname).write_bytes(await upload.read())

    try:
        geom = json.loads((await geometry.read()).decode("utf-8"))
    except Exception:
        raise HTTPException(status_code=400, detail="invalid geometry json")
    (dir_path / "geometry.json").write_text(json.dumps(geom, indent=2))

    await db.mark_refined(sub_id)
    return {"ok": True, "id": sub_id, "status": "refined"}
