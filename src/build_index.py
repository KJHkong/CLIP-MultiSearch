import os
import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
import clip         # OpenAI的CLIP模型
from PIL import Image
from tqdm import tqdm
import faiss        # Facebook的向量相似性搜索库

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}   # 支持的图像扩展名集合

#图像文件扫描函数
def list_images(root: Path) -> List[Path]:
    paths = []
    for p in root.rglob("*"):   # rglob: 递归匹配所有文件和目录
        if p.is_file() and p.suffix.lower() in IMG_EXTS:      #如果p是文件且扩展名在支持列表中
            paths.append(p)
    return sorted(paths)   # 排序确保顺序一致

def main(
    data_dir: str = "data/images",
    out_dir: str = "storage",
    model_name: str = "ViT-B/32",
    batch_size: int = 32,
):
    device = "cpu"
    model, preprocess = clip.load(model_name, device=device)
    model.eval()  # 设置为评估模式（关闭dropout等训练特性）

    data_root = Path(data_dir)
    out_root = Path(out_dir)

    # 创建输出目录（parents=True允许创建多级目录）
    out_root.mkdir(parents=True, exist_ok=True)

    # 扫描所有图像文件
    img_paths = list_images(data_root)

     # 检查是否找到图像
    if not img_paths:
        raise RuntimeError(f"No images found under: {data_root.resolve()}")


    #=============批量编码循环=============
    
    #存储所有特征向量
    all_feats = [] 
    global_id = 0 #记录图片对应的id
    #创建元数据文件（JSON Lines格式，每行一个JSON对象）
    meta_path = out_root / "meta.jsonl"
    with meta_path.open("w", encoding="utf-8") as fmeta:  #写入模式

        # 批量处理图像（tqdm显示进度条）,控制台进度条的描述文字是 “Encoding images”
        for i in tqdm(range(0, len(img_paths), batch_size), desc="Encoding images"):

            # 1. 获取当前批次
            batch = img_paths[i:i+batch_size]  # 切片获取批次，batch里面放的是该批次的图片对应的路径

            imgs = []  # 存储预处理后的图像张量(转为RGB格式后放入)
            metas: List[Dict] = []  # 存储元数据

            # 2. 遍历批次中的每个图像
            for p in batch:  #p是一个个路径
                try:
                    # 打开图像并转换为RGB（确保三通道）  单张图片转张量（形状：[3, 224, 224]）
                    img = preprocess(Image.open(p).convert("RGB"))
                    imgs.append(img)

                    # 记录元数据
                    metas.append({
                        "id": global_id,  # 唯一ID
                        "type": "image",    # 类型标识
                        "path": str(p.as_posix())   # 文件路径
                    })
                    global_id += 1
                except Exception as e:
                    # 跳过无法读取的图像（损坏文件等）
                    continue

            # 3. 检查当前批次是否有有效图像
            if not imgs:
                continue

            #=============特征提取过程===============

            # 将图像列表堆叠为批次张量 [batch_size, 3, 224, 224]
            # 比如处理图片时，单张图片张量形状是 [3, 224, 224]（通道数 × 高度 × 宽度），
            imgs_tensor = torch.stack(imgs).to(device)

            # 关闭梯度计算，减少内存占用
            with torch.no_grad():
                # 提取图像特征 [batch_size, embedding_dim]
                feats = model.encode_image(imgs_tensor).float()

                # 特征归一化（单位向量）    
                # 归一化后，向量的点积 = 余弦相似度
                feats = feats / feats.norm(dim=-1, keepdim=True)
            
            # 转换为NumPy数组，FAISS需要float32
            feats_np = feats.cpu().numpy().astype("float32")

            # ==============保存元数据和特征=================
            # 写入元数据到JSON Lines文件
            for m in metas:
                fmeta.write(json.dumps(m, ensure_ascii=False) + "\n")

            # 收集所有批次特征
            all_feats.append(feats_np)

    # ==========FAISS索引构建===============
    # 合并所有批次的特征 [num_images, embedding_dim]
    feats = np.concatenate(all_feats, axis=0)
    dim = feats.shape[1]   #embedding的维度,特征维度，如512

    # 创建FAISS索引（内积索引，因为特征已归一化）
    # IndexFlatIP: 使用内积（点积）进行相似度计算
    # 由于向量已归一化，内积 = 余弦相似度
    index = faiss.IndexFlatIP(dim)

    '''
    # 假设feats是一个3×512的矩阵：
    feats = np.array([
    [0.1, 0.2, ..., 0.5],  # 图片1的特征
    [0.3, 0.1, ..., 0.2],  # 图片2的特征
    [0.9, 0.0, ..., 0.1],  # 图片3的特征
    ])

    # index.add(feats) 做了两件事：
    # 1. 存储这些向量
    # 2. 建立搜索数据结构（对于IndexFlatIP，就是简单的存储）
    '''

    # 添加特征到索引
    index.add(feats)

    # 保存索引到文件
    faiss.write_index(index, str(out_root / "index.faiss"))

    print(f"✅ Done. Indexed {index.ntotal} items. Saved to {out_root.resolve()}")

if __name__ == "__main__":
    main()
