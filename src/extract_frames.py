"""
从 data/videos 下的视频中按固定时间间隔抽取关键帧，保存帧图到 storage/frames，并返回元数据列表。
"""
from pathlib import Path
from typing import List, Dict

import cv2
from PIL import Image
from tqdm import tqdm

VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".webm"}


def list_videos(root: Path) -> List[Path]:
    paths = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            paths.append(p)
    return sorted(paths)


def extract_frames(
    video_dir: str = "data/videos",
    out_dir: str = "storage/frames",
    interval_sec: float = 2.0,
) -> List[Dict]:
    """
    扫描 video_dir 下所有视频，每隔 interval_sec 秒抽一帧，保存到 out_dir，并返回元数据列表。

    返回：List[Dict]，每项含 frame_path, video_path, timestamp_sec
    """
    video_root = Path(video_dir)
    frames_root = Path(out_dir)
    frames_root.mkdir(parents=True, exist_ok=True)

    video_paths = list_videos(video_root)
    if not video_paths:
        raise RuntimeError(f"No videos found under: {video_root.resolve()}")

    meta_list: List[Dict] = []
    for vp in tqdm(video_paths, desc="Extracting frames"):
        cap = cv2.VideoCapture(str(vp))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_interval = max(1, int(fps * interval_sec))
        frame_idx = 0
        saved_count = 0
        stem = vp.stem

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                ts = frame_idx / fps
                fname = f"{stem}_{saved_count:04d}_{ts:.1f}s.jpg"
                fpath = frames_root / fname
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                Image.fromarray(rgb).save(str(fpath))
                meta_list.append({
                    "frame_path": str(fpath.as_posix()),
                    "video_path": str(vp.as_posix()),
                    "timestamp_sec": round(ts, 2),
                })
                saved_count += 1
            frame_idx += 1
        cap.release()

    return meta_list


if __name__ == "__main__":
    frames_meta = extract_frames(video_dir="data/videos", out_dir="storage/frames", interval_sec=2.0)
    print(f"Extracted {len(frames_meta)} frames.")
