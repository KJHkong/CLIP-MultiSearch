"""
对 data/videos 下的视频抽关键帧，用 CLIP 编码后追加到现有 FAISS 索引和 meta.jsonl。
需先运行 build_index.py 建立图片索引，再运行本脚本追加视频帧。
"""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import json
from typing import List, Dict

import numpy as np
import torch
import clip
from PIL import Image
from tqdm import tqdm
import faiss

from src.extract_frames import extract_frames


def main(
    video_dir: str = "data/videos",
    frames_dir: str = "storage/frames",
    storage_dir: str = "storage",
    model_name: str = "ViT-B/32",
    interval_sec: float = 2.0,
    batch_size: int = 32,
):
    device = "cpu"
    storage = Path(storage_dir)
    index_path = storage / "index.faiss"
    meta_path = storage / "meta.jsonl"

    if not index_path.exists() or not meta_path.exists():
        raise RuntimeError(
            f"Index not found. Run 'python src/build_index.py' first to build image index."
        )

    print("1. Extracting frames from videos...")
    frames_meta = extract_frames(
        video_dir=video_dir,
        out_dir=frames_dir,
        interval_sec=interval_sec,
    )
    if not frames_meta:
        raise RuntimeError("No frames extracted. Check video_dir and video files.")

    print(f"2. Loading CLIP and existing index (extracted {len(frames_meta)} frames)...")
    model, preprocess = clip.load(model_name, device=device)
    model.eval()
    index = faiss.read_index(str(index_path))
    next_id = index.ntotal

    all_feats = []
    new_meta_lines: List[str] = []

    for i in tqdm(range(0, len(frames_meta), batch_size), desc="Encoding video frames"):
        batch = frames_meta[i : i + batch_size]
        imgs = []
        metas = []

        for m in batch:
            try:
                img = preprocess(Image.open(m["frame_path"]).convert("RGB"))
                imgs.append(img)
                metas.append({
                    "id": next_id,
                    "type": "video_frame",
                    "path": m["frame_path"],
                    "video_path": m["video_path"],
                    "timestamp_sec": m["timestamp_sec"],
                })
                next_id += 1
            except Exception:
                continue

        if not imgs:
            continue

        imgs_tensor = torch.stack(imgs).to(device)
        with torch.no_grad():
            feats = model.encode_image(imgs_tensor).float()
            feats = feats / feats.norm(dim=-1, keepdim=True)
        feats_np = feats.cpu().numpy().astype("float32")
        all_feats.append(feats_np)
        for m in metas:
            new_meta_lines.append(json.dumps(m, ensure_ascii=False))

    if not all_feats:
        raise RuntimeError("No valid frames to encode.")

    feats = np.concatenate(all_feats, axis=0)
    index.add(feats)
    faiss.write_index(index, str(index_path))

    with meta_path.open("a", encoding="utf-8") as f:
        for line in new_meta_lines:
            f.write(line + "\n")

    print(f"✅ Done. Added {len(new_meta_lines)} video frames. Total index size: {index.ntotal}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Append video keyframes to existing FAISS index")
    p.add_argument("--video_dir", default="data/videos", help="Directory containing videos")
    p.add_argument("--frames_dir", default="storage/frames", help="Output dir for extracted frames")
    p.add_argument("--storage_dir", default="storage", help="Index and meta location")
    p.add_argument("--model_name", default="ViT-B/32", help="CLIP model (must match build_index)")
    p.add_argument("--interval_sec", type=float, default=2.0, help="Frame interval in seconds")
    p.add_argument("--batch_size", type=int, default=32)
    args = p.parse_args()
    main(
        video_dir=args.video_dir,
        frames_dir=args.frames_dir,
        storage_dir=args.storage_dir,
        model_name=args.model_name,
        interval_sec=args.interval_sec,
        batch_size=args.batch_size,
    )
