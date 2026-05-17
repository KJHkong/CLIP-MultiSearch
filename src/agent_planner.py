"""
Planner Agent：用 DeepSeek V4 Pro 将复杂 Query 拆解为可执行的子查询计划。
参考 VSA (Vision Search Assistant) 的 what/how/by what to search 框架。
"""
import json
from dataclasses import dataclass, field
from typing import List, Optional
from openai import OpenAI

from src.config import LLM_API_BASE, LLM_API_KEY, LLM_MODEL
from src.tools_schema import Modality


@dataclass
class SubQuery:
    """单个子查询。"""
    text: str                                       # 子查询文本（英文、偏视觉化）
    modality: str = "image"                         # "image" | "video" | "both"
    weight: float = 1.0                             # 融合权重
    rationale: str = ""                             # 为什么拆出这个子查询


@dataclass
class PlanOutput:
    """Planner 输出：完整的检索计划。"""
    query_type: str = "simple"                      # simple / complex / comparative / sequential / cross_modal
    sub_queries: List[SubQuery] = field(default_factory=list)
    plan_summary: str = ""                          # 规划思路简述
    fusion_strategy: str = "max"                    # max / weighted / llm_rerank


# ---------- Planner System Prompt ----------

PLANNER_SYSTEM = """You are a multimodal search planner. Your job is to analyze a user's search query and decompose it into executable sub-queries.

Given a user query (in Chinese or English), output a JSON object with this exact structure:
{
  "query_type": "simple" | "complex" | "comparative" | "sequential" | "cross_modal",
  "plan_summary": "one short sentence explaining your plan in Chinese",
  "sub_queries": [
    {
      "text": "a short English visual description for CLIP image search",
      "modality": "image" | "video" | "both",
      "weight": 1.0,
      "rationale": "why this sub-query is needed, in Chinese"
    }
  ],
  "fusion_strategy": "max" | "weighted" | "llm_rerank"
}

RULES:
- "query_type": "simple" if the query is a single visual concept (e.g. "a cat", "红色汽车"). For simple queries, output exactly ONE sub_query.
- "complex" if the query has multiple constraints or attributes (e.g. "a red car parked near a blue building").
- "sequential" if the query mentions a time sequence or order (e.g. "先有猫再出现狗", "after the presenter shows a chart").
- "cross_modal" if the query references both images and videos simultaneously.
- "comparative" if the query asks to compare things.
- Each sub_query.text MUST be a short English visual description (under 15 words), suitable for CLIP text-to-image search.
- For sequential queries, order sub_queries by time order.
- Assign weight based on importance: core queries get 1.0, auxiliary queries get 0.5-0.7.
- Use "llm_rerank" fusion when there are 3+ sub_queries from different modalities; otherwise "max".
- Output ONLY the JSON object, no markdown fences, no extra text."""


def plan_query(user_query: str) -> PlanOutput:
    """
    分析用户查询，输出结构化检索计划。

    如果查询简单到不需要 LLM（例如单词/短语级别），直接返回 simple 透传计划。
    复杂查询则调用 DeepSeek V4 Pro 拆解。
    """
    q = user_query.strip()
    if not q:
        return PlanOutput(query_type="simple", sub_queries=[], plan_summary="空查询")

    # 快速判断：短于 15 个词且不含复杂关键词 → 直接透传
    words = q.split()
    simple_keywords = ["和", "与", "或者", "然后", "之后", "之前", "比较", "区别", "对比",
                       "and", "or", "then", "after", "before", "compare", "versus", "vs"]
    is_simple = len(words) <= 8 and not any(kw in q.lower() for kw in simple_keywords)

    if is_simple:
        return PlanOutput(
            query_type="simple",
            sub_queries=[SubQuery(text=q, modality="both", weight=1.0, rationale="简单查询，直接检索")],
            plan_summary="简单查询，单步检索",
            fusion_strategy="max",
        )

    # 复杂查询 → 调 LLM 拆解
    if not LLM_API_KEY:
        # 无 API Key 时降级为简单透传
        return PlanOutput(
            query_type="simple",
            sub_queries=[SubQuery(text=q, modality="both", weight=1.0, rationale="无 LLM，直接检索")],
            plan_summary="无 LLM API Key，降级为单步检索",
            fusion_strategy="max",
        )

    try:
        client = OpenAI(base_url=LLM_API_BASE, api_key=LLM_API_KEY)
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": PLANNER_SYSTEM},
                {"role": "user", "content": f"User query: {q}\n\nOutput JSON:"},
            ],
            max_tokens=800,
            temperature=0.1,  # 低温度保证输出稳定
        )
        raw = (response.choices[0].message.content or "").strip()
        return _parse_plan(raw, q)
    except Exception as e:
        print(f"[Planner] LLM 调用失败，降级为简单透传: {e}")
        return PlanOutput(
            query_type="simple",
            sub_queries=[SubQuery(text=q, modality="both", weight=1.0, rationale=f"LLM 失败降级: {str(e)[:80]}")],
            plan_summary="LLM 失败，降级为单步检索",
            fusion_strategy="max",
        )


def _parse_plan(raw: str, fallback_query: str) -> PlanOutput:
    """解析 LLM 输出的 JSON，含容错。"""
    # 清理可能的 markdown fence
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # 尝试提取第一个 JSON 对象
        brace_start = raw.find("{")
        brace_end = raw.rfind("}")
        if brace_start != -1 and brace_end > brace_start:
            try:
                data = json.loads(raw[brace_start:brace_end + 1])
            except json.JSONDecodeError:
                return _fallback_simple(fallback_query)
        else:
            return _fallback_simple(fallback_query)

    sub_queries = []
    for sq in data.get("sub_queries", []):
        sub_queries.append(SubQuery(
            text=sq.get("text", fallback_query),
            modality=sq.get("modality", "both"),
            weight=float(sq.get("weight", 1.0)),
            rationale=sq.get("rationale", ""),
        ))

    if not sub_queries:
        sub_queries = [SubQuery(text=fallback_query, modality="both", weight=1.0, rationale="解析无子查询，回退")]

    return PlanOutput(
        query_type=data.get("query_type", "simple"),
        sub_queries=sub_queries,
        plan_summary=data.get("plan_summary", ""),
        fusion_strategy=data.get("fusion_strategy", "max"),
    )


def _fallback_simple(query: str) -> PlanOutput:
    return PlanOutput(
        query_type="simple",
        sub_queries=[SubQuery(text=query, modality="both", weight=1.0, rationale="JSON 解析失败，回退透传")],
        plan_summary="解析失败，降级为单步检索",
        fusion_strategy="max",
    )


# ---------- 模态路由器 ----------

def route_and_search(plan: PlanOutput, tool_registry) -> List:
    """
    根据 Plan 中的 sub_queries，调用 ToolRegistry 中的工具执行检索。
    返回 List[ToolResult]。
    """
    all_results = []
    for sq in plan.sub_queries:
        modality = sq.modality
        try:
            if modality == "image":
                tr = tool_registry.call("search_image_by_text", query=sq.text)
            elif modality == "video":
                tr = tool_registry.call("search_video_by_text", query=sq.text)
            elif modality == "both":
                # 搜图 + 搜视频，两者都拿
                tr_img = tool_registry.call("search_image_by_text", query=sq.text)
                tr_vid = tool_registry.call("search_video_by_text", query=sq.text)
                all_results.append(tr_img)
                all_results.append(tr_vid)
                continue
            else:
                tr = tool_registry.call("search_image_by_text", query=sq.text)
            all_results.append(tr)
        except Exception as e:
            print(f"[Router] 工具调用失败 ({sq.modality}:{sq.text[:40]}): {e}")
            continue

    return all_results


# ---------- 结果融合 ----------

@dataclass
class FusedResults:
    """多路检索结果融合后的统一输出。"""
    results: List                     # List[SearchResult]，去重+排序后的最终结果
    source_count: int                 # 融合前总结果数
    fusion_method: str                # 使用的融合策略
    per_tool_summaries: List[str] = field(default_factory=list)  # 各工具的摘要


def fuse_results(
    tool_results: List,
    plan: PlanOutput,
    topk: int = 20,
) -> FusedResults:
    """
    融合多个 ToolResult：去重 + 加权 + 重排序 → FusedResults。
    """
    # 1. 按 path 去重，保留最高分
    best: dict = {}
    total = 0
    summaries = []
    for tr in tool_results:
        total += len(tr.results)
        summaries.append(tr.summary())
        for sr in tr.results:
            key = sr.path
            if key not in best or sr.score > best[key].score:
                best[key] = sr

    # 2. 按分数降序
    ranked = sorted(best.values(), key=lambda x: x.score, reverse=True)

    # 3. 截断
    ranked = ranked[:topk]

    return FusedResults(
        results=ranked,
        source_count=total,
        fusion_method=plan.fusion_strategy,
        per_tool_summaries=summaries,
    )


def llm_rerank(
    fused: FusedResults,
    user_query: str,
    topk: int = 10,
) -> FusedResults:
    """
    用 DeepSeek V4 Pro 对融合后的结果做语义重排序。
    将 top-N 结果的路径/描述给 LLM，让它根据 query 语义重新排序。
    """
    if not LLM_API_KEY or len(fused.results) <= 3:
        return fused

    # 构建候选列表供 LLM 评判
    candidates_text = ""
    for i, sr in enumerate(fused.results[:20]):
        ts_info = f" [timestamp={sr.timestamp_sec}s]" if sr.timestamp_sec else ""
        candidates_text += f"[{i}] {sr.path}{ts_info} score={sr.score:.4f}\n"

    prompt = f"""User query: {user_query}

Below are search results for the query. Re-rank them by semantic relevance to the query.
Return ONLY the indices of the top {topk} results in order, as a JSON array like: [3, 0, 7, 1, ...]

{candidates_text}"""

    try:
        client = OpenAI(base_url=LLM_API_BASE, api_key=LLM_API_KEY)
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a search result re-ranker. Output ONLY a JSON array of indices."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=200,
            temperature=0.0,
        )
        raw = (response.choices[0].message.content or "").strip()
        # 解析索引数组
        if raw.startswith("["):
            indices = json.loads(raw)
        else:
            # 尝试提取
            start = raw.find("[")
            end = raw.rfind("]")
            indices = json.loads(raw[start:end + 1]) if start != -1 else []

        # 按 LLM 给出的顺序重排
        id_to_sr = {i: sr for i, sr in enumerate(fused.results[:20])}
        reranked = []
        seen = set()
        for idx in indices:
            if isinstance(idx, int) and idx in id_to_sr and idx not in seen:
                reranked.append(id_to_sr[idx])
                seen.add(idx)
        # 补充未覆盖的
        for i, sr in enumerate(fused.results[:20]):
            if i not in seen:
                reranked.append(sr)
                seen.add(i)

        fused.results = reranked[:topk]
        fused.fusion_method = "llm_rerank"
    except Exception as e:
        print(f"[Rerank] LLM 重排序失败: {e}")

    return fused


# ========== 简易测试 ==========
if __name__ == "__main__":
    test_queries = [
        "一只猫",                                          # simple
        "一只坐在红色沙发上的黑猫",                          # complex
        "先展示一张猫的图片，然后是包含公式的白板视频片段",    # sequential
        "红色背景下的蓝色杯子，且有人在旁边",                 # complex
    ]
    for q in test_queries:
        plan = plan_query(q)
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        print(f"  Type: {plan.query_type}")
        print(f"  Plan: {plan.plan_summary}")
        for sq in plan.sub_queries:
            print(f"  → [{sq.modality}] {sq.text} (w={sq.weight})")
