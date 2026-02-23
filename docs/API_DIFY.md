# CLIP-MultiSearch API (for Dify Agent)

This document describes how to run the API and configure it as a **Dify custom API tool**.

---

## 1. Install dependencies

```bash
pip install fastapi "uvicorn[standard]" python-multipart
# or
pip install -r requirements.txt
```

---

## 2. Run the API server

From the **project root** (`my_CLIP/`):

```bash
# Option A: uvicorn (recommended)
python -m uvicorn src.api_fastapi:app --host 0.0.0.0 --port 8000

# Option B: direct
python src/api_fastapi.py
```

- API base URL: `http://localhost:8000` (or `http://<your-ip>:8000` if Dify runs in Docker on another machine).
- Interactive docs: `http://localhost:8000/docs`.

---

## 3. Endpoints summary

| GET | `/` | Service info |
| GET | `/files/{path}` | Serve image/video files under `data/images`, `storage/frames`, `storage/clips`, `data/videos` (for direct display) |
| POST | `/search/text` | Text-to-image search (JSON body) |
| POST | `/search/video` | Text-to-video keyframe search (JSON body) |
| POST | `/search/image` | Image-to-image search (multipart: file + `topk`) |
| POST | `/search/image/json` | Image-to-image search (form: `image_base64` + `topk`) |

---

## 4. Dify Tool configuration

In Dify: **Studio → Tools → Create Tool → Custom API**.

### 4.1 Text search tool

- **Name**: `clip_text_search`
- **URL**: `http://<API_HOST>:8000/search/text`
- **Method**: POST
- **Headers**: `Content-Type: application/json`
- **Body** (JSON):

```json
{
  "query": "{{query}}",
  "topk": 10,
  "expand_n": 6,
  "use_llm": true
}
```

- **Parameters** (add in Dify):
  - `query` (string, required): search query
  - `topk` (number, optional): default 10
  - `expand_n` (number, optional): default 6
  - `use_llm` (boolean, optional): default true

Response shape:

```json
{
  "success": true,
  "message": "ok",
  "results": [
    { "path": "...", "score": 0.85, "type": "image" }
  ],
  "expand_source": "LLM"
}
```

### 4.1.1 让 Agent 在回复里带出图片链接（Dify 对话内展示）

若希望 Agent 在文字回复中**直接写出可点击的图片**，请把系统提示词里关于图片的部分改成下面这段（强调必须逐条输出 Markdown）：

```text
当用户要搜图、找图时，使用 search_text 工具，把用户的意思填进 query。拿到 results 后：
1. 先简短总结（例如：找到了 N 张相关图片）。
2. 若 results 里有 image_url 字段，你必须在前几条（最多 5 条）中，每条单独一行写出 Markdown 图片，格式严格为：![图片](对应的image_url)，不要省略、不要用“如下”代替，直接写出完整 URL。例如：
![图片1](http://host.docker.internal:8000/files/data/images/xxx.jpg)
![图片2](http://host.docker.internal:8000/files/data/images/yyy.jpg)
若没有 image_url 则只做文字总结。
```

注意：Dify 的「调试与预览」对话界面**不一定支持把 Markdown 渲染成内嵌图片**，可能只显示为链接。若仍看不到图，可复制 Agent 回复里的链接到浏览器打开，或使用「访问 API」提供的 Web 应用/API 在自有前端中渲染图片。

### 4.2 Video keyframe search tool

- **Name**: `clip_video_search`
- **URL**: `http://<API_HOST>:8000/search/video`
- **Method**: POST
- **Headers**: `Content-Type: application/json`
- **Body**:

```json
{
  "query": "{{query}}",
  "topk": 5,
  "expand_n": 6,
  "use_llm": true
}
```

- **Parameters**: `query` (required), `topk`, `expand_n`, `use_llm` (same as above).

Response:

```json
{
  "success": true,
  "message": "ok",
  "results": [
    {
      "path": "...",
      "score": 0.28,
      "type": "video_frame",
      "video_path": "data/videos/xxx.mp4",
      "timestamp_sec": 16.0,
      "clip_path": "/abs/path/storage/clips/xxx_160000_2.0s.mp4"
    }
  ],
  "expand_source": "LLM"
}
```

### 4.3 Image search tool (for “search by image”)

If Dify sends a **file** (image upload):

- **URL**: `http://<API_HOST>:8000/search/image`
- **Method**: POST
- **Body**: multipart/form-data
  - `image`: file
  - `topk`: number (e.g. 10)

If Dify sends **base64** in a JSON/form parameter:

- **URL**: `http://<API_HOST>:8000/search/image/json`
- **Method**: POST
- **Body**: form-data
  - `image_base64`: string (base64 or `data:image/jpeg;base64,...`)
  - `topk`: number

Configure in Dify according to whether your workflow uses “file” or “variable (base64)”.

---

## 5. Network note (Dify in Docker)

If Dify runs inside Docker and the API runs on the host:

- Use `host.docker.internal:8000` (Windows/Mac) or the host’s LAN IP (e.g. `192.168.x.x:8000`) as `<API_HOST>`.
- Ensure the host firewall allows port 8000.

---

## 6. Quick test (curl)

```bash
# Text search
curl -X POST http://localhost:8000/search/text \
  -H "Content-Type: application/json" \
  -d '{"query":"a cat","topk":3}'

# Video search
curl -X POST http://localhost:8000/search/video \
  -H "Content-Type: application/json" \
  -d '{"query":"小猫","topk":2}'
```
