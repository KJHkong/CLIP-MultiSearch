from typing import List   #这是为了给列表做类型标注，不导入也没关系

def expand_query(user_query: str, n: int = 6) -> List[str]:
    """
    MVP版本：无需LLM，规则模板扩展 + 中英混合兜底
    返回多条prompt，用于分别检索再融合。

    参数：
    - user_query: 用户输入的查询文本
    - n: 最多返回的查询数量（默认6条）
    
    返回值：
    - List[str]: 扩展后的查询列表

    """

    # 去除输入查询的前后空格
    q = user_query.strip()

    # 初始化查询列表
    prompts = []

    # 1. 保留原始查询（第一原则：用户说的最重要）
    if q:
        prompts.append(q)

    # 原理：CLIP支持多语言，中文也能直接处理
    # 例子：用户输入"海军陆战队士兵" → 直接加入列表

    # 2. 英文模板扩展（提升CLIP理解效果）
    templates = [
        "a photo of {}",    # 基础模板
        "a high quality photo of {}",   # 高质量图片模板
        "a close-up photo of {}",    # 特写图片模板 
        "an image of {}",
        "a photograph of {}",
    ]
    # 这些模板是根据CLIP训练数据的统计特征设计的
    # CLIP在训练时看到大量类似"a photo of ..."的描述

    # 将用户查询填充到每个模板中
    for t in templates:
        prompts.append(t.format(q))

    # 例子：q="soldier" → 
    #   "a photo of soldier"
    #   "a high quality photo of soldier"

    # 去重 + 截断
    uniq = []   # 存储去重后的结果
    seen = set()  # 用于快速检查重复的集合

     # 遍历所有生成的查询
    for p in prompts:
        # 如果还没见过这个查询且不为空
        if p not in seen and p:
            uniq.append(p)  # 加入结果列表
            seen.add(p)   # 加入已见集合

    # 返回前n个结果（最多n个）
    return uniq[:n]      
