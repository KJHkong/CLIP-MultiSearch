"""
Reflection Agent：用 DeepSeek V4 Pro 对检索结果做自我批判，判断是否需要重试。
参考 ViDoRAG 的 iterative reasoning 和 DeepImageSearch 的自主探索思路。
"""
import json
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from openai import OpenAI

from src.config import LLM_API_BASE, LLM_API_KEY, LLM_MODEL
from src.tools_schema import ToolResult


@dataclass
class ReflectionOutput:
    """Reflection 模块输出：对当前检索质量的判断。"""
    sufficient: bool                                # 结果是否充分
    confidence: float                               # 置信度 0.0–1.0
    missing_info: str = ""                          # 缺失什么信息
    next_query: Optional[str] = None                # 建议的下一轮查询
    reason: str = ""                                # 判断理由


# ---------- Search Memory ----------

@dataclass
class RoundRecord:
    """单轮搜索记录。"""
    round_num: int
    query: str
    result_summary: str                             # 结果摘要
    reflection: Optional[ReflectionOutput] = None


class SearchMemory:
    """
    搜索记忆：记录每轮的查询、结果和反思判断。
    防止重复搜索，为下一轮 Query 改写提供上下文。
    """

    def __init__(self):
        self.rounds: List[RoundRecord] = []

    def add_round(self, query: str, tool_results: List[ToolResult]) -> RoundRecord:
        """记录一轮搜索。"""
        # 生成结果摘要
        summaries = [tr.summary() for tr in tool_results]
        summary = " | ".join(summaries) if summaries else "无结果"
        record = RoundRecord(
            round_num=len(self.rounds) + 1,
            query=query,
            result_summary=summary[:300],
        )
        self.rounds.append(record)
        return record

    def record_reflection(self, reflection: ReflectionOutput):
        """将反思结果绑定到最近一轮。"""
        if self.rounds:
            self.rounds[-1].reflection = reflection

    def get_history(self) -> List[Dict[str, Any]]:
        """获取搜索历史摘要，供 Planner/Reflector 使用。"""
        return [
            {
                "round": r.round_num,
                "query": r.query,
                "summary": r.result_summary,
                "sufficient": r.reflection.sufficient if r.reflection else None,
                "confidence": r.reflection.confidence if r.reflection else None,
            }
            for r in self.rounds
        ]

    def avoid_redundancy(self, new_query: str) -> bool:
        """检查新 query 是否与历史高度重复。简单规则：完全相同或重叠度高。"""
        new_set = set(new_query.lower().split())
        for r in self.rounds:
            old_set = set(r.query.lower().split())
            if new_set == old_set:
                return True
            overlap = len(new_set & old_set) / max(len(new_set | old_set), 1)
            if overlap > 0.85:
                return True
        return False


# ---------- Reflector System Prompt ----------

REFLECTOR_SYSTEM = """You are a search quality evaluator. Given a user's original question, the current search results, and the search history, evaluate whether the results are sufficient to answer the question.

Output a JSON object with this exact structure:
{
  "sufficient": true or false,
  "confidence": 0.0 to 1.0,
  "missing_info": "if not sufficient, what key information is missing? in Chinese",
  "next_query": "if not sufficient, a refined search query to find the missing information. Should be a short English visual description suitable for CLIP. null if sufficient.",
  "reason": "one short sentence in Chinese explaining your judgment"
}

RULES:
- confidence >= 0.7 AND sufficient=true → stop searching
- confidence < 0.6 → definitely need to retry with a better query
- If the top results clearly match what the user is looking for, set sufficient=true
- If top scores are very low (< 0.25 for CLIP cosine similarity), results are probably unrelated
- next_query should be different from previous queries in the history to avoid redundancy
- Focus on what VISUAL evidence is still missing, not just text-level matching
- Output ONLY the JSON object, no markdown fences, no extra text."""


def reflect(
    user_query: str,
    tool_results: List[ToolResult],
    memory: SearchMemory,
) -> ReflectionOutput:
    """
    让 LLM 判断当前检索结果是否充分回答用户问题。

    参数：
        user_query: 用户原始查询
        tool_results: 当前轮所有工具的返回结果
        memory: 搜索记忆（含历史轮次）

    返回：
        ReflectionOutput: 反思判断
    """
    # 快速规则判断：如果所有工具都没有结果，直接返回 insufficient
    if not tool_results or all(len(tr.results) == 0 for tr in tool_results):
        return ReflectionOutput(
            sufficient=False,
            confidence=0.0,
            missing_info="无搜索结果",
            next_query=user_query,  # 先用原 query 重试
            reason="所有工具均未返回结果",
        )

    # 快速规则判断：如果有高分结果（>0.35），可能已足够
    all_scores = [sr.score for tr in tool_results for sr in tr.results if sr.score > 0]
    if all_scores and max(all_scores) > 0.40:
        # 给出一个初步的高置信度，但仍让 LLM 做最终判断
        pass

    # 构建结果摘要供 LLM 评估
    results_text = ""
    for tr in tool_results:
        results_text += f"\n--- {tr.tool_name} ---\n"
        for i, sr in enumerate(tr.results[:5]):
            ts = f" [ts={sr.timestamp_sec}s]" if sr.timestamp_sec else ""
            results_text += f"  [{i}] {sr.path}{ts} score={sr.score:.4f}\n"

    history_text = ""
    for h in memory.get_history():
        history_text += f"Round {h['round']}: query='{h['query']}' sufficient={h['sufficient']} confidence={h['confidence']}\n"

    prompt = f"""Original user question: {user_query}

Search history:
{history_text or '(no history)'}

Current search results:
{results_text}

Evaluate: Are these results sufficient to answer the user's question?"""

    if not LLM_API_KEY:
        return _rule_based_reflect(user_query, tool_results, memory)

    try:
        client = OpenAI(base_url=LLM_API_BASE, api_key=LLM_API_KEY)
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": REFLECTOR_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=400,
            temperature=0.1,
        )
        raw = (response.choices[0].message.content or "").strip()
        return _parse_reflection(raw, user_query)
    except Exception as e:
        print(f"[Reflector] LLM 调用失败，使用规则判断: {e}")
        return _rule_based_reflect(user_query, tool_results, memory)


def _parse_reflection(raw: str, fallback_query: str) -> ReflectionOutput:
    """解析 LLM 输出的 JSON，含容错。"""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        brace_start = raw.find("{")
        brace_end = raw.rfind("}")
        if brace_start != -1 and brace_end > brace_start:
            try:
                data = json.loads(raw[brace_start:brace_end + 1])
            except json.JSONDecodeError:
                return ReflectionOutput(
                    sufficient=True, confidence=0.5,
                    reason="JSON 解析失败，默认终止", next_query=None,
                )
        else:
            return ReflectionOutput(
                sufficient=True, confidence=0.5,
                reason="JSON 解析失败，默认终止", next_query=None,
            )

    return ReflectionOutput(
        sufficient=data.get("sufficient", True),
        confidence=float(data.get("confidence", 0.5)),
        missing_info=data.get("missing_info", ""),
        next_query=data.get("next_query") if not data.get("sufficient") else None,
        reason=data.get("reason", ""),
    )


def _rule_based_reflect(
    user_query: str,
    tool_results: List[ToolResult],
    memory: SearchMemory,
) -> ReflectionOutput:
    """无 LLM 时的规则回退判断。"""
    all_scores = [sr.score for tr in tool_results for sr in tr.results if sr.score > 0]

    if not all_scores:
        return ReflectionOutput(sufficient=False, confidence=0.0, missing_info="无结果",
                                next_query=user_query, reason="无搜索命中")

    max_score = max(all_scores)
    avg_score = sum(all_scores) / len(all_scores)
    round_count = len(memory.rounds)

    if max_score > 0.35:
        return ReflectionOutput(sufficient=True, confidence=max_score, reason="高分结果命中")
    elif round_count >= 3:
        return ReflectionOutput(sufficient=True, confidence=0.4, reason="已达最大重试次数")
    elif max_score > 0.25:
        return ReflectionOutput(sufficient=False, confidence=0.4, missing_info="分数偏低",
                                next_query=user_query, reason="分数偏低，建议改写查询重试")
    else:
        return ReflectionOutput(sufficient=False, confidence=0.2, missing_info="分数过低",
                                next_query=user_query, reason="分数过低，建议放宽查询条件")


# ========== 简易测试 ==========
if __name__ == "__main__":
    from src.tools_schema import SearchResult

    # 模拟数据
    mem = SearchMemory()
    tr = ToolResult(
        tool_name="search_image_by_text",
        query="a black cat on a red sofa",
        results=[
            SearchResult(id=0, modality="image", path="images/cat1.jpg", score=0.32),
            SearchResult(id=1, modality="image", path="images/cat2.jpg", score=0.28),
        ],
    )

    mem.add_round("a black cat on a red sofa", [tr])
    ref = reflect("Find a black cat sitting on a red sofa", [tr], mem)
    print(f"  Sufficient: {ref.sufficient}, Confidence: {ref.confidence}")
    print(f"  Reason: {ref.reason}")
    if ref.next_query:
        print(f"  Next: {ref.next_query}")
