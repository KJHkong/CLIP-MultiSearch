import json
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
import clip
import faiss

# 导入之前写的查询扩展器
from src.query_expand import expand_query

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
