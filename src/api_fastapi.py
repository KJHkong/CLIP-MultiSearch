"""
CLIP-MultiSearch FastAPI: expose text / image / video search for Dify Agent tools.
Run from project root: python -m uvicorn src.api_fastapi:app --host 0.0.0.0 --port 8000
"""
import os
import sys
import base64
import tempfile
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from typing import List, Optional
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.search import search_with_fusion, search_by_image, search_video_clips

# 供 Dify/前端展示图片用的 API 根地址（Dify 在 Docker 时设为 http://host.docker.internal:8000）
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000").rstrip("/")
ALLOWED_FILE_PREFIXES = ("data/images/", "data/videos/", "storage/frames/", "storage/clips/")

app = FastAPI(
    title="CLIP-MultiSearch API",
    description="Text / image / video search for Dify Agent tools",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Request / Response models ----------

class TextSearchRequest(BaseModel):
    query: str = Field(..., description="Natural language search query (e.g. 'a cat', '小猫')")
    topk: int = Field(10, ge=1, le=50, description="Number of results to return")
    expand_n: int = Field(6, ge=1, le=10, description="Number of expanded prompts for fusion")
    use_llm: bool = Field(True, description="Use LLM to rewrite query (recommended for Chinese)")


class ImageSearchRequest(BaseModel):
    topk: int = Field(10, ge=1, le=50, description="Number of results to return")
    # image_url or image_base64 can be set by Dify; we also support file upload in another endpoint


class VideoSearchRequest(BaseModel):
    query: str = Field(..., description="Natural language search query for video keyframes")
    topk: int = Field(5, ge=1, le=20, description="Number of video keyframe results")
    expand_n: int = Field(6, ge=1, le=10, description="Number of expanded prompts")
    use_llm: bool = Field(True, description="Use LLM to rewrite query")


# Generic result item (image or video_frame)
class SearchResultItem(BaseModel):
    path: str
    score: float
    type: Optional[str] = "image"
    video_path: Optional[str] = None
    timestamp_sec: Optional[float] = None
    clip_path: Optional[str] = None
    # 可访问的 URL，便于 Dify/前端直接展示图片或视频
    image_url: Optional[str] = None
    video_url: Optional[str] = None


class TextSearchResponse(BaseModel):
    success: bool = True
    message: str = "ok"
    results: List[SearchResultItem]
    expand_source: str = ""


class VideoSearchResponse(BaseModel):
    success: bool = True
    message: str = "ok"
    results: List[SearchResultItem]
    expand_source: str = ""


def _item_from_result(r: dict) -> SearchResultItem:
    path = r.get("path", "")
    item_type = r.get("type", "image")
    clip_path = r.get("clip_path")
    image_url = None
    video_url = None
    # 为图片生成可访问 URL；视频则优先用 clip 的 URL
    if item_type == "image" and path:
        norm = path.replace("\\", "/").lstrip("/")
        if any(norm.startswith(p) for p in ALLOWED_FILE_PREFIXES):
            image_url = f"{API_BASE_URL}/files/{norm}"
    elif item_type == "video_frame":
        if clip_path:
            norm = Path(clip_path).as_posix()
            if "storage/clips" in norm:
                video_url = f"{API_BASE_URL}/files/storage/clips/{Path(clip_path).name}"
        if not video_url and path:
            norm = path.replace("\\", "/").lstrip("/")
            if norm.startswith("data/videos/"):
                video_url = f"{API_BASE_URL}/files/{norm}"
    return SearchResultItem(
        path=path,
        score=round(float(r.get("score", 0)), 4),
        type=item_type,
        video_path=r.get("video_path"),
        timestamp_sec=r.get("timestamp_sec"),
        clip_path=clip_path,
        image_url=image_url,
        video_url=video_url,
    )


# ---------- Endpoints ----------

@app.get("/")
def root():
    return {"service": "CLIP-MultiSearch API", "endpoints": ["/search/text", "/search/image", "/search/video"], "files_base": "/files"}


@app.get("/health")
def health():
    """Health check for Dify or liveness probes."""
    return {"status": "ok", "service": "CLIP-MultiSearch API", "version": "0.1.0"}


@app.get("/files/{file_path:path}")
def serve_file(file_path: str):
    """提供图片/视频文件访问，便于 Dify 或前端直接展示。仅允许 data/images、storage/frames、storage/clips、data/videos 下的路径。"""
    norm = file_path.replace("\\", "/").lstrip("/")
    if not any(norm.startswith(p) for p in ALLOWED_FILE_PREFIXES):
        raise HTTPException(status_code=403, detail="path not allowed")
    full = (_root / norm).resolve()
    if not full.is_file() or _root not in full.parents:
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(full)


@app.post("/search/text", response_model=TextSearchResponse)
def search_text(req: TextSearchRequest):
    """Text-to-image (and video frame) search. Use for natural language queries."""
    if not (req.query or req.query.strip()):
        raise HTTPException(status_code=400, detail="query is required")
    try:
        results, _, expand_source = search_with_fusion(
            user_query=req.query.strip(),
            topk=req.topk,
            expand_n=req.expand_n,
            use_llm=req.use_llm,
        )
        items = [_item_from_result(r) for r in results]
        return TextSearchResponse(results=items, expand_source=expand_source)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/video", response_model=VideoSearchResponse)
def search_video(req: VideoSearchRequest):
    """Text-to-video keyframe search. Returns ranked video clips with paths and timestamps."""
    if not (req.query or req.query.strip()):
        raise HTTPException(status_code=400, detail="query is required")
    try:
        results, _, expand_source = search_video_clips(
            user_query=req.query.strip(),
            topk=req.topk,
            expand_n=req.expand_n,
            use_llm=req.use_llm,
        )
        items = [_item_from_result(r) for r in results]
        return VideoSearchResponse(results=items, expand_source=expand_source)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/image", response_model=TextSearchResponse)
async def search_image_upload(
    image: UploadFile = File(...),
    topk: int = Form(10),
):
    """Image-to-image search. Upload an image file (e.g. image/jpeg, image/png)."""
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (e.g. image/jpeg, image/png)")
    try:
        raw = await image.read()
        with tempfile.NamedTemporaryFile(suffix=Path(image.filename or "img").suffix or ".jpg", delete=False) as f:
            f.write(raw)
            path = f.name
        try:
            results = search_by_image(image_input=path, topk=topk)
            items = [_item_from_result(r) for r in results]
            return TextSearchResponse(results=items, expand_source="image")
        finally:
            Path(path).unlink(missing_ok=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/image/json", response_model=TextSearchResponse)
def search_image_base64(
    image_base64: str = Form(..., description="Base64-encoded image data (e.g. data:image/jpeg;base64,... or raw base64)"),
    topk: int = Form(10),
):
    """Image-to-image search with base64 image (for Dify tool that sends JSON body)."""
    try:
        if "," in image_base64:
            image_base64 = image_base64.split(",", 1)[1]
        raw = base64.b64decode(image_base64)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(raw)
            path = f.name
        try:
            results = search_by_image(image_input=path, topk=topk)
            items = [_item_from_result(r) for r in results]
            return TextSearchResponse(results=items, expand_source="image")
        finally:
            Path(path).unlink(missing_ok=True)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image or search failed: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
