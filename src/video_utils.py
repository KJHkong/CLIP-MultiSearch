import subprocess
from pathlib import Path
from typing import Optional

import cv2

# 优先用 imageio-ffmpeg 自带的 ffmpeg，无需单独安装、配置 PATH
def _get_ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def make_video_clip(
    video_path: str,
    center_ts: float,
    clip_len: float = 2.0,
    out_dir: str = "storage/clips",
    fps_fallback: float = 25.0,
) -> Optional[str]:
    """
    从原视频中截取以 center_ts 为中心、长度约 clip_len 秒的一小段，保存为新视频，返回新视频绝对路径。
    优先用 ffmpeg（H.264，浏览器兼容好），失败则用 OpenCV。
    """
    video_p = Path(video_path).resolve()
    if not video_p.exists():
        return None

    clips_root = Path(out_dir).resolve()
    clips_root.mkdir(parents=True, exist_ok=True)

    start_ts = max(0.0, center_ts - clip_len / 2)
    clip_name = f"{video_p.stem}_{int(center_ts*100):06d}_{clip_len:.1f}s.mp4"
    clip_path = clips_root / clip_name

    # 1. 优先用 ffmpeg，显式输出 H.264+AAC 确保浏览器可播放（-c copy 可能保留不兼容编码）
    ffmpeg_exe = _get_ffmpeg_exe()
    try:
        cmd = [
            ffmpeg_exe, "-y",
            "-ss", str(start_ts),
            "-i", str(video_p),
            "-t", str(clip_len),
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",  # 便于流式播放
            str(clip_path),
        ]
        r = subprocess.run(cmd, capture_output=True, timeout=60)
        if r.returncode == 0 and clip_path.exists():
            return str(clip_path)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # 2. 回退到 OpenCV
    cap = cv2.VideoCapture(str(video_p))
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or fps_fallback
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        cap.release()
        return None

    center_frame = int(center_ts * fps)
    half = int(clip_len * fps / 2)
    start_frame = max(0, center_frame - half)
    end_frame = min(total_frames - 1, center_frame + half)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        return None

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(clip_path), fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current = start_frame
    while current <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
        current += 1

    writer.release()
    cap.release()

    if clip_path.exists():
        return str(clip_path.resolve())
    return None

