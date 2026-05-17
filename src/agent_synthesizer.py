"""
Answer Synthesis Agent：用 DeepSeek V4 Pro 融合多轮搜索结果和视觉证据，
生成带结构化引用的最终答案。
"""
import json
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from openai import OpenAI

from src.config import LLM_API_BASE, LLM_API_KEY, LLM_MODEL


@dataclass
class Citation:
    """单条引用。"""
    evidence_id: str                              # E1, E2, ...
    asset_display: str                            # 显示用：路径或帧时间戳
    note: str                                     # 简短注释


@dataclass
class SearchTrajectoryStep:
    """搜索轨迹中的一步。"""
    round: int
    plan_summary: str
    results_count: int
    reflection_reason: str = ""


@dataclass
class SynthesisOutput:
    """Answer Synthesis 完整输出。"""
    answer: str                                   # 最终答案文本
    citations: List[Citation] = field(default_factory=list)
    search_trajectory: List[SearchTrajectoryStep] = field(default_factory=list)
    confidence: float = 0.0
    disclaimer: str = ""                          # 如果信息不足，诚实说明


# ---------- Synthesizer System Prompt ----------

SYNTHESIZER_SYSTEM = """You are a multimodal search summarizer. Given a user's question, the search trajectory (multiple rounds of retrieval), and the visual evidence gathered, generate a comprehensive answer.

Rules:
1. Answer the user's question directly and clearly, in Chinese (or the same language as the query).
2. After key claims, cite the evidence using [E1], [E2], etc.
3. If evidence items contradict each other, note the contradiction.
4. If information is insufficient to fully answer, honestly state what's missing.
5. Keep the answer concise but complete (3-8 sentences).

Output a JSON object:
{
  "answer": "your answer text with [E1][E2] citations embedded",
  "confidence": 0.0 to 1.0,
  "disclaimer": "if information is insufficient, explain what's missing. Otherwise empty string."
}

Output ONLY the JSON object, no extra text."""


def synthesize(
    user_query: str,
    search_trajectory: List[SearchTrajectoryStep],
    evidence_summaries: List[Dict[str, Any]],
) -> SynthesisOutput:
    """
    基于搜索轨迹和证据摘要，生成带引用的最终答案。

    参数：
        user_query: 用户原始问题
        search_trajectory: 每轮搜索的步骤摘要
        evidence_summaries: 证据列表，每项含 evidence_id, visual_description, rationale 等
    """
    if not evidence_summaries:
        return SynthesisOutput(
            answer="未找到与查询相关的视觉证据。",
            confidence=0.0,
            disclaimer="所有检索工具均未返回有效结果",
        )

    # 构建证据文本
    evidence_text = ""
    for ev in evidence_summaries:
        ts = f" [timestamp={ev.get('timestamp_sec')}s]" if ev.get('timestamp_sec') else ""
        evidence_text += (
            f"[{ev.get('evidence_id', '?')}] modality={ev.get('modality', 'image')}"
            f" path={ev.get('asset_path', '')}{ts}\n"
            f"    visual: {ev.get('visual_description', '')}\n"
            f"    rationale: {ev.get('grounding_rationale', '')}\n\n"
        )

    # 构建轨迹文本
    traj_text = ""
    for step in search_trajectory:
        traj_text += f"Round {step.round}: {step.plan_summary} ({step.results_count} results) — {step.reflection_reason}\n"

    prompt = f"""User question: {user_query}

Search trajectory:
{traj_text}

Evidence gathered:
{evidence_text}

Generate a comprehensive answer with citations."""

    if not LLM_API_KEY:
        return _rule_based_synthesize(user_query, evidence_summaries, search_trajectory)

    try:
        client = OpenAI(base_url=LLM_API_BASE, api_key=LLM_API_KEY)
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYNTHESIZER_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=800,
            temperature=0.3,
        )
        raw = (response.choices[0].message.content or "").strip()
        data = _parse_synthesis_json(raw)

        # 构建 Citation 列表
        citations = []
        for ev in evidence_summaries:
            eid = ev.get("evidence_id", "?")
            asset = ev.get("asset_path", "")
            ts = ev.get("timestamp_sec")
            if ts:
                asset = f"frame {ts:.1f}s"
            citations.append(Citation(
                evidence_id=eid,
                asset_display=asset[:60],
                note=ev.get("visual_description", "")[:80],
            ))

        return SynthesisOutput(
            answer=data.get("answer", "无法生成答案"),
            citations=citations,
            search_trajectory=search_trajectory,
            confidence=float(data.get("confidence", 0.5)),
            disclaimer=data.get("disclaimer", ""),
        )

    except Exception as e:
        print(f"[Synthesizer] LLM 调用失败: {e}")
        return _rule_based_synthesize(user_query, evidence_summaries, search_trajectory)


def _parse_synthesis_json(raw: str) -> dict:
    """解析 Synthesizer LLM 输出的 JSON。"""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
        if raw.endswith("```"):
            raw = raw[:-3]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        brace_start = raw.find("{")
        brace_end = raw.rfind("}")
        if brace_start != -1 and brace_end > brace_start:
            try:
                return json.loads(raw[brace_start:brace_end + 1])
            except json.JSONDecodeError:
                pass
    return {"answer": raw, "confidence": 0.5, "disclaimer": ""}


def _rule_based_synthesize(
    user_query: str,
    evidence_summaries: List[Dict],
    trajectory: List[SearchTrajectoryStep],
) -> SynthesisOutput:
    """无 LLM 时的规则回退合成。"""
    parts = [f"搜索了 {len(trajectory)} 轮，找到 {len(evidence_summaries)} 条相关证据："]
    for ev in evidence_summaries[:5]:
        eid = ev.get("evidence_id", "?")
        desc = ev.get("visual_description", "") or ev.get("grounding_rationale", "")
        parts.append(f"[{eid}] {desc}")
    answer = "\n".join(parts) if len(parts) > 1 else "未找到充分证据"

    citations = [
        Citation(evidence_id=ev.get("evidence_id", "?"), asset_display=ev.get("asset_path", "")[:60], note="")
        for ev in evidence_summaries[:5]
    ]
    return SynthesisOutput(answer=answer, citations=citations, search_trajectory=trajectory, confidence=0.3)


# ========== 简易测试 ==========
if __name__ == "__main__":
    traj = [
        SearchTrajectoryStep(round=1, plan_summary="直接检索", results_count=10, reflection_reason="分数偏低"),
        SearchTrajectoryStep(round=2, plan_summary="放宽查询重试", results_count=8, reflection_reason="命中高分结果"),
    ]
    evs = [
        {"evidence_id": "E1", "modality": "image", "asset_path": "images/cat.jpg",
         "visual_description": "A black cat sitting on a red sofa", "grounding_rationale": "完美匹配查询"},
    ]
    result = synthesize("Find a black cat on a red sofa", traj, evs)
    print(f"  Answer: {result.answer}")
    print(f"  Confidence: {result.confidence}")
    print(f"  Citations: {len(result.citations)}")
    print(f"  Trajectory steps: {len(result.search_trajectory)}")
