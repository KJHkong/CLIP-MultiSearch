# 本阶段改进总结（面试准备用）

## 一、视频关键帧检索与播放

### 1. 功能实现
- **search_video_clips**：基于文本在 FAISS 中检索视频帧（`type=video_frame`），对每个命中关键帧调用 `make_video_clip` 生成 2 秒短视频片段
- **候选池扩大**：视频帧在索引中占比小，将 `fetch_k` 从 `topk*3` 调整为 `max(topk*80, 500)`，确保能筛出足够视频帧
- **路径解析**：`video_path` 可能为相对路径，在 `search.py` 和 `ui_gradio.py` 中统一基于项目根解析，避免从不同目录运行时路径错误

### 2. 视频片段生成（video_utils.py）
- **ffmpeg 优先**：显式输出 H.264 + AAC，确保浏览器可播放（`-c copy` 可能保留不兼容编码）
- **imageio-ffmpeg**：优先使用 `imageio_ffmpeg.get_ffmpeg_exe()`，无需单独安装 ffmpeg、配置 PATH，pip 安装即可
- **回退逻辑**：ffmpeg 失败时回退到 OpenCV（mp4v），clip 生成失败时回退到原视频路径

### 3. Gradio 视频播放修复
- **allowed_paths**：`demo.launch(allowed_paths=[str(_root)])`，否则 Gradio 无法访问项目目录下的本地视频文件
- **绝对路径**：传给 `gr.Video` 的路径统一转为绝对路径

### 4. UI 布局优化
- **等高布局**：`gr.Row(equal_height=True)`，左侧 Radio 与右侧 Video 等高
- **固定视频高度**：`gr.Video(height=360)`，避免右侧过高、左侧留白

---

## 二、涉及文件

| 文件 | 改动 |
|------|------|
| `src/search.py` | `search_video_clips`、video_path 项目根解析 |
| `src/video_utils.py` | `make_video_clip`、imageio-ffmpeg、H.264 编码 |
| `src/ui_gradio.py` | 视频 Tab、allowed_paths、路径解析、equal_height、Video height |
| `requirements.txt` | 新增 `imageio-ffmpeg` |

---

## 三、面试可强调点

1. **多模态统一索引**：图片与视频帧共用同一 CLIP 特征空间和 FAISS 索引，检索逻辑统一
2. **工程鲁棒性**：ffmpeg 可选（imageio-ffmpeg 兜底）、路径解析、Gradio 文件访问策略
3. **用户体验**：视频片段播放、布局优化、LLM 改写支持中文查询
