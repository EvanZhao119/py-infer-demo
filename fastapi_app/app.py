# fastapi_app/app.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Tuple
from PIL import Image
import io
import base64

from infer import predict

app = FastAPI(title="ResNet Inference Service (FastAPI)")

class B64ImageRequest(BaseModel):
    image_base64: str
    topk: int = 5

def _read_image_from_upload(file: UploadFile) -> Image.Image:
    data = file.file.read()
    return Image.open(io.BytesIO(data)).convert("RGB")

def _read_image_from_b64(b64str: str) -> Image.Image:
    data = base64.b64decode(b64str)
    return Image.open(io.BytesIO(data)).convert("RGB")

@app.post("/predict/file")
async def predict_file(file: UploadFile = File(...), topk: int = 5):
    try:
        img = _read_image_from_upload(file)
        res = predict(img, topk=topk)
        return JSONResponse({"topk": [{"label": l, "prob": p} for l, p in res]})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/b64")
async def predict_b64(req: B64ImageRequest):
    try:
        img = _read_image_from_b64(req.image_base64)
        res = predict(img, topk=req.topk)
        return JSONResponse({"topk": [{"label": l, "prob": p} for l, p in res]})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}
