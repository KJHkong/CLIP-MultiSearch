"""
LLM 查询改写：将用户输入（中文或英文）扩展为多条英文、偏视觉的短句，用于 CLIP 检索。
使用硅基流动 DeepSeek API（OpenAI 兼容）。
"""
from typing import List

from openai import OpenAI

from src.config import LLM_API_BASE, LLM_API_KEY, LLM_MODEL


def rewrite_for_search(user_query: str, n: int = 6) -> List[str]:
    """
    用 LLM 将用户查询改写成 n 条英文、视觉化的短句，供 CLIP 检索使用。
    无论用户输入中文还是英文，都统一生成多条英文描述以提升检索效果。

    参数：
        user_query: 用户输入的查询（中文或英文）
        n: 期望生成的 prompt 数量

    返回：
        长度不超过 n 的英文短句列表

    异常：
        未配置 API Key、网络错误或模型返回异常时会抛出，由调用方做规则兜底。
    """
    q = user_query.strip()
    if not q:
        return []

    if not LLM_API_KEY:
        raise ValueError("未设置 SILICONFLOW_API_KEY，请在环境变量中配置硅基流动 API Key")

    client = OpenAI(base_url=LLM_API_BASE, api_key=LLM_API_KEY)

    system_prompt = """You are a helper for image search. Given a user's search query (in Chinese or English), output exactly N short English phrases that describe the same scene in a visual way, suitable for matching images. Each phrase should be one line, concise (under 15 words), and focus on objects, scenes, and atmosphere. Do not number the lines, do not add explanations, only output the phrases, one per line."""

    user_prompt = f"""Generate exactly {n} short English phrases for image search. User query: {q}

Output {n} lines, each line one phrase:"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=400,
        temperature=0.3,
    )
    content = (response.choices[0].message.content or "").strip()
    # 按行解析，去掉空行和行内编号/空格
    lines = [line.strip().lstrip("0123456789.)\t- ").strip() for line in content.splitlines() if line.strip()]
    # 去重且保留顺序，截断到 n 条
    seen = set()
    out = []
    for line in lines:
        if line and line not in seen and len(line) <= 200:
            seen.add(line)
            out.append(line)
            if len(out) >= n:
                break
    if not out:
        # 解析失败时至少返回原查询
        out = [q]
    return out[:n]
