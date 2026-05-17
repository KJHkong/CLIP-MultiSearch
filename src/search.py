import json
from pathlib import Path
from typing import List, Dict, Tuple, Union

import numpy as np
import torch
import clip
import faiss
from PIL import Image

# 导入之前写的查询扩展器
from src.query_expand import expand_query
from src.video_utils import make_video_clip
from src.tools_schema import ToolResult, wrap_as_tool_result

def load_meta(meta_path: Path) -> List[Dict]:

    """
    加载元数据文件（JSON Lines格式）
    
    参数：
    - meta_path: 元数据文件路径
    
    返回：
    - List[Dict]: 每个图片的元数据字典列表
    """

    items = []
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))  # 解析每行的JSON
    return items

def encode_text(model, device: str, text: str) -> np.ndarray:

    """
    将文本编码为CLIP特征向量（已归一化）
    
    参数：
    - model: CLIP模型
    - device: 计算设备（CPU/GPU）
    - text: 要编码的文本
    
    返回：
    - np.ndarray: 形状为 (1, 512) 的特征向量
    """
    # 1. 分词：CLIP有自己的tokenizer
    tokens = clip.tokenize([text]).to(device)

    # 2. 编码文本
    with torch.no_grad(): # 关闭梯度计算
        feat = model.encode_text(tokens).float()  # [1, 512]

        # 3. 归一化（单位向量）
        feat = feat / feat.norm(dim=-1, keepdim=True)

     # 4. 转换为NumPy数组（FAISS需要）
    return feat.cpu().numpy().astype("float32")  # (1, dim)


def _image_to_pil(image_input: Union[str, np.ndarray]) -> Image.Image:
    """将路径或 numpy 数组转为 PIL Image（RGB）。"""
    if isinstance(image_input, str):
        return Image.open(image_input).convert("RGB")
    if isinstance(image_input, np.ndarray):
        return Image.fromarray(image_input.astype(np.uint8)).convert("RGB")
    raise TypeError("image_input 应为文件路径(str)或 numpy 数组(H,W,3)")


def search_by_image(
    image_input: Union[str, np.ndarray],
    topk: int = 20,
    model_name: str = "ViT-B/32",
    storage_dir: str = "storage",
) -> List[Dict]:
    """
    以图搜图：用一张图片的 CLIP 特征在 FAISS 索引中检索最相似的图片。
    与 build_index 使用相同的预处理和模型，保证向量空间一致。

    参数：
        image_input: 图片路径(str)或 numpy 数组(H,W,3)，如 Gradio Image 组件返回值
        topk: 返回的结果数量
        model_name: CLIP 模型名，需与建索引时一致
        storage_dir: 索引与 meta 所在目录

    返回：
        与 search_with_fusion 相同结构的 results：List[Dict]，每项含 path, score, type 等
    """
    device = "cpu"
    model, preprocess = clip.load(model_name, device=device)
    model.eval()

    storage = Path(storage_dir)
    index = faiss.read_index(str(storage / "index.faiss"))
    meta = load_meta(storage / "meta.jsonl")

    pil_img = _image_to_pil(image_input)
    img_tensor = preprocess(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = model.encode_image(img_tensor).float()
        feat = feat / feat.norm(dim=-1, keepdim=True)
    q = feat.cpu().numpy().astype("float32")  # (1, dim)

    scores, ids = index.search(q, topk)
    scores = scores[0]
    ids = ids[0]

    results = []
    for idx, s in zip(ids, scores):
        if idx < 0 or idx >= len(meta):
            continue
        item = meta[idx].copy()
        item["score"] = float(s)
        results.append(item)
    return results


def search_with_fusion(
    user_query: str,
    topk: int = 20,
    expand_n: int = 6,
    use_llm: bool = True,
    fusion: str = "max",
    model_name: str = "ViT-B/32",
    storage_dir: str = "storage",
) -> Tuple[List[Dict], List[Tuple[str, float]], str]:
    
    """
    主搜索函数：扩展查询 + 融合搜索结果
    
    参数：
    - user_query: 用户查询文本
    - topk: 最终返回的结果数量
    - expand_n: 扩展的查询数量
    - use_llm: 是否用 LLM 改写/扩展查询（中英文均可用）；False 则用规则模板
    - fusion: 融合策略（"max"或"vote"）
    - model_name: CLIP模型名称
    - storage_dir: 索引和元数据存储目录
    
    返回：
    - Tuple[搜索结果列表, 扩展查询调试信息, 扩展方式说明字符串]
    """
    # 1. 加载CLIP模型
    device = "cpu"
    model, _ = clip.load(model_name, device=device)  # 不需要preprocess
    model.eval()  # 评估模式

    # 2. 加载FAISS索引和元数据
    storage = Path(storage_dir)
    index = faiss.read_index(str(storage / "index.faiss"))  # 读取向量索引
    meta = load_meta(storage / "meta.jsonl")  # 读取图片元数据

    # 3. 扩展查询：一个变多个（LLM 或规则）
    prompts, expand_source = expand_query(user_query, n=expand_n, use_llm=use_llm)

    # 4. 记录每个prompt的最高分 
    per_prompt_scores = []  # 存储 (prompt, 最高分) 的列表

    # 5. 融合用字典：记录每个图片在不同prompt下的最高分
    best_score = {}  # key: 图片ID, value: 在所有prompt中的最高相似度

    # 6. 遍历每个扩展查询
    for p in prompts:
        # 6.1 将文本编码为特征向量
        q = encode_text(model, device, p)   # (1, 512)

        # 6.2 在FAISS索引中搜索
        # scores: 相似度分数矩阵 [1, topk]
        # ids: 对应的图片ID矩阵 [1, topk]
        scores, ids = index.search(q, topk)  

        # 取第一行（因为只有一个查询）
        scores = scores[0]  # 形状: (topk,)
        ids = ids[0]  # 形状: (topk,)

        # 6.3 记录这个prompt的top1分数（调试用）
        per_prompt_scores.append((p, float(scores[0])))

        # 6.4 更新best_score字典（max融合策略）
        for idx, s in zip(ids, scores):
            if idx < 0 or idx >= len(meta) :  # FAISS可能返回-1表示无效结果并添加边界检查
                continue
            s = float(s)  # 转换为Python float

            # 核心融合逻辑：取最高分
            # 如果这个图片还没记录，或者当前分数更高，就更新
            if (idx not in best_score) or (s > best_score[idx]):
                best_score[idx] = s

    # 7. 按分数排序，取topk个结果
    # best_score.items() → [(id1, score1), (id2, score2), ...]
    # sorted(..., key=lambda x: x[1], reverse=True) → 按分数降序排序
    # [:topk] → 取前topk个
    ranked = sorted(best_score.items(), key=lambda x: x[1], reverse=True)[:topk]

    # 8. 构建最终结果列表
    results = []
    for idx, s in ranked:
        # 复制元数据（避免修改原数据）
        item = meta[idx].copy()
        item["score"] = s  # 添加分数信息
        results.append(item)

    # 9. 返回结果、调试信息和扩展方式（便于界面显示是否用了 LLM）
    return results, per_prompt_scores, expand_source


def search_video_clips(
    user_query: str,
    topk: int = 5,
    expand_n: int = 6,
    use_llm: bool = True,
    model_name: str = "ViT-B/32",
    storage_dir: str = "storage",
    clip_len: float = 2.0,
) -> Tuple[List[Dict], List[Tuple[str, float]], str]:
    """
    基于文本查询视频关键帧，并为每个关键帧生成一小段视频片段。
    逻辑：先用 search_with_fusion 做统一检索，然后从结果中筛选 type=video_frame 的项，
    对前 topk 个关键帧截取中心在 timestamp_sec 附近的 clip_len 秒视频。
    """
    # 先跑一遍统一检索（图片 + 视频帧），需要足够多的候选才能筛出视频帧
    # 索引中视频帧占比小，topk*3 容易全是图片；改为取 500+ 条再筛
    fetch_k = max(topk * 80, 500)
    all_results, prompt_scores, expand_source = search_with_fusion(
        user_query=user_query,
        topk=fetch_k,
        expand_n=expand_n,
        use_llm=use_llm,
        model_name=model_name,
        storage_dir=storage_dir,
    )

    # 只保留视频帧
    video_results = [r for r in all_results if r.get("type") == "video_frame"]
    video_results = video_results[:topk]

    # 为每个关键帧生成 clip_path（video_path 可能为相对路径，需基于项目根解析）
    proj_root = Path(__file__).resolve().parents[1]
    for r in video_results:
        vpath = r.get("video_path")
        if vpath and not Path(vpath).is_absolute():
            vpath = str((proj_root / vpath).resolve())
        ts = float(r.get("timestamp_sec", 0.0))
        clip_path = make_video_clip(
            video_path=vpath,
            center_ts=ts,
            clip_len=clip_len,
            out_dir="storage/clips",
        )
        r["clip_path"] = clip_path

    return video_results, prompt_scores, expand_source


# ========== ToolResult 包装函数（供 Agent 调用） ==========

def tool_search_image_by_text(
    query: str,
    topk: int = 20,
    expand_n: int = 6,
    use_llm: bool = True,
) -> ToolResult:
    """Tool 包装版：文本搜图，返回标准 ToolResult。"""
    results, _, expand_source = search_with_fusion(
        user_query=query, topk=topk, expand_n=expand_n, use_llm=use_llm,
    )
    return wrap_as_tool_result(
        results, tool_name="search_image_by_text", query=query,
        metadata={"expand_source": expand_source},
    )


def tool_search_video_by_text(
    query: str,
    topk: int = 5,
    expand_n: int = 6,
    use_llm: bool = True,
) -> ToolResult:
    """Tool 包装版：文本搜视频帧，返回标准 ToolResult。"""
    results, _, expand_source = search_video_clips(
        user_query=query, topk=topk, expand_n=expand_n, use_llm=use_llm,
    )
    return wrap_as_tool_result(
        results, tool_name="search_video_by_text", query=query,
        metadata={"expand_source": expand_source},
    )


def tool_search_image_by_image(
    image_path: str,
    topk: int = 20,
) -> ToolResult:
    """Tool 包装版：以图搜图，返回标准 ToolResult。"""
    results = search_by_image(image_input=image_path, topk=topk)
    return wrap_as_tool_result(
        results, tool_name="search_image_by_image", query=f"image:{image_path}",
    )
