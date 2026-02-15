from typing import List, Tuple  # 用于类型标注


def expand_query_rules(user_query: str, n: int = 6) -> List[str]:
    """
    规则模板扩展：无需 LLM，用英文模板 + 原始查询生成多条 prompt。
    用于兜底或关闭「使用 LLM 改写」时。
    """
    q = user_query.strip()
    prompts = []
    if q:
        prompts.append(q)
    templates = [
        "a photo of {}",
        "a high quality photo of {}",
        "a close-up photo of {}",
        "an image of {}",
        "a photograph of {}",
    ]
    for t in templates:
        prompts.append(t.format(q))
    uniq = []
    seen = set()
    for p in prompts:
        if p not in seen and p:
            uniq.append(p)
            seen.add(p)
    return uniq[:n]


def expand_query(user_query: str, n: int = 6, use_llm: bool = True) -> Tuple[List[str], str]:
    """
    统一入口：扩展查询为多条 prompt，用于多路检索再融合。
    返回 (prompts, source_info)，source_info 用于界面显示当前是 LLM 还是规则、以及失败原因。
    """
    if not use_llm:
        return expand_query_rules(user_query, n=n), "规则（未勾选 LLM）"
    try:
        from src.llm_rewrite import rewrite_for_search
        prompts = rewrite_for_search(user_query, n=n)
        return prompts, "LLM"
    except Exception as e:
        err_msg = str(e)[:120]  # 截断过长错误
        print("[query_expand] LLM 失败，已退回规则扩展:", err_msg)
        return expand_query_rules(user_query, n=n), f"规则（LLM 失败: {err_msg}）"      
