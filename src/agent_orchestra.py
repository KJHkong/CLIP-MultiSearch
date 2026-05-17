"""
Agent Orchestra — 主编排器：将 Planner、Router、Reflector、Evidence Extractor、
Synthesizer 串联为完整的 Agentic Multimodal Search Pipeline。

Pipeline:
  Plan → Route → Search → Reflect → (Loop) → Evidence → Synthesize → AgentResponse

参考 ViDoRAG (2025) 的 iterative reasoning 范式。
"""
import time
import json
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from src.config import LLM_API_KEY, VLM_API_KEY
from src.tools_schema import ToolResult, ToolRegistry, SearchResult
from src.agent_planner import (
    PlanOutput, SubQuery, plan_query, route_and_search,
    fuse_results, FusedResults, llm_rerank,
)
from src.agent_reflector import (
    ReflectionOutput, SearchMemory, reflect,
)
from src.agent_evidence import (
    Evidence, extract_all_evidences,
)
from src.agent_synthesizer import (
    SynthesisOutput, Citation, SearchTrajectoryStep,
    synthesize,
)
from src.search import (
    tool_search_image_by_text,
    tool_search_video_by_text,
    tool_search_image_by_image,
)


# ---------- Agent Response ----------

@dataclass
class AgentResponse:
    """Agent 完整搜索响应。"""
    answer: str
    citations: List[Citation] = field(default_factory=list)
    search_trajectory: List[SearchTrajectoryStep] = field(default_factory=list)
    evidences: List[Dict[str, Any]] = field(default_factory=list)
    fused_results: List[Dict[str, Any]] = field(default_factory=list)
    total_rounds: int = 0
    confidence: float = 0.0
    disclaimer: str = ""
    elapsed_ms: float = 0.0
    plan: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": [
                {"evidence_id": c.evidence_id, "asset_display": c.asset_display, "note": c.note}
                for c in self.citations
            ],
            "search_trajectory": [
                {"round": s.round, "plan_summary": s.plan_summary,
                 "results_count": s.results_count, "reflection_reason": s.reflection_reason}
                for s in self.search_trajectory
            ],
            "evidences": self.evidences,
            "fused_results": self.fused_results,
            "total_rounds": self.total_rounds,
            "confidence": self.confidence,
            "disclaimer": self.disclaimer,
            "elapsed_ms": self.elapsed_ms,
            "plan": self.plan,
        }


# ---------- Tool Registry 工厂 ----------

def build_default_registry() -> ToolRegistry:
    """构建默认工具注册中心，包含所有检索工具。"""
    registry = ToolRegistry()
    from src.tools_schema import ToolDefinition, Modality

    registry.register(ToolDefinition(
        name="search_image_by_text",
        description="用自然语言描述搜索图片。输入英文视觉描述，返回最匹配的图片。",
        modality=Modality.IMAGE,
        func=tool_search_image_by_text,
        input_schema={"query": "string (English visual description)", "topk": "int (default 20)"},
    ))
    registry.register(ToolDefinition(
        name="search_video_by_text",
        description="用自然语言描述搜索视频关键帧。返回视频片段及时间戳。",
        modality=Modality.VIDEO,
        func=tool_search_video_by_text,
        input_schema={"query": "string (English visual description)", "topk": "int (default 5)"},
    ))
    registry.register(ToolDefinition(
        name="search_image_by_image",
        description="以图搜图：上传一张图片，搜索视觉相似的图片。",
        modality=Modality.IMAGE,
        func=tool_search_image_by_image,
        input_schema={"image_path": "string (path to image file)", "topk": "int (default 20)"},
    ))
    return registry


# ---------- 主编排器 ----------

class AgentOrchestra:
    """
    Agentic Multimodal Search 主编排器。

    使用方式:
        orchestra = AgentOrchestra()
        response = orchestra.search("一只坐在红色沙发上的黑猫")
        print(response.answer)
    """

    def __init__(
        self,
        tool_registry: Optional[ToolRegistry] = None,
        max_rounds: int = 3,
        confidence_threshold: float = 0.6,
        enable_evidence: bool = True,
        enable_llm_rerank: bool = True,
        evidence_top_n: int = 5,
    ):
        self.tool_registry = tool_registry or build_default_registry()
        self.max_rounds = max_rounds
        self.confidence_threshold = confidence_threshold
        self.enable_evidence = enable_evidence
        self.enable_llm_rerank = enable_llm_rerank
        self.evidence_top_n = evidence_top_n

    def search(self, user_query: str) -> AgentResponse:
        """
        执行完整的 Agentic 搜索流程。

        参数:
            user_query: 用户原始查询（中文或英文）

        返回:
            AgentResponse: 含答案、引用、搜索轨迹的完整响应
        """
        t0 = time.perf_counter()
        memory = SearchMemory()
        all_tool_results: List[ToolResult] = []
        trajectory_steps: List[SearchTrajectoryStep] = []
        final_plan: Optional[PlanOutput] = None

        # ===== Phase 1-3: Iterative Search Loop =====
        for rnd in range(self.max_rounds):
            # --- Step 1: Plan ---
            if rnd == 0:
                plan = plan_query(user_query)
            else:
                # 用 Reflection 建议的 next_query 重新规划
                plan = plan_query(reflection.next_query or user_query)

            final_plan = plan

            # --- Step 2: Route & Search ---
            tool_results = route_and_search(plan, self.tool_registry)
            all_tool_results.extend(tool_results)

            # --- Step 3: Fuse ---
            fused = fuse_results(tool_results, plan, topk=20)
            if self.enable_llm_rerank and len(fused.results) > 3:
                fused = llm_rerank(fused, user_query, topk=15)

            # --- Step 4: Reflect ---
            reflection = reflect(user_query, tool_results, memory)
            memory.add_round(
                query=plan.sub_queries[0].text if plan.sub_queries else user_query,
                tool_results=tool_results,
            )
            memory.record_reflection(reflection)

            trajectory_steps.append(SearchTrajectoryStep(
                round=rnd + 1,
                plan_summary=plan.plan_summary,
                results_count=len(fused.results),
                reflection_reason=reflection.reason,
            ))

            if reflection.sufficient and reflection.confidence >= self.confidence_threshold:
                break

        # 最终融合所有轮次结果
        final_fused = fuse_results(all_tool_results, final_plan, topk=20)

        # ===== Phase 4: Evidence Grounding =====
        evidences: List[Evidence] = []
        evidence_dicts: List[Dict[str, Any]] = []
        if self.enable_evidence and VLM_API_KEY:
            evidences = extract_all_evidences(final_fused, user_query, top_n=self.evidence_top_n)
            for ev in evidences:
                evidence_dicts.append({
                    "evidence_id": ev.evidence_id,
                    "modality": ev.modality,
                    "asset_path": ev.asset_path,
                    "timestamp_sec": ev.timestamp_sec,
                    "video_path": ev.video_path,
                    "clip_path": ev.clip_path,
                    "relevance_score": ev.relevance_score,
                    "visual_description": ev.visual_description,
                    "grounding_rationale": ev.grounding_rationale,
                    "bounding_hint": ev.bounding_hint,
                })

        # ===== Phase 5: Answer Synthesis =====
        synthesis = synthesize(
            user_query=user_query,
            search_trajectory=trajectory_steps,
            evidence_summaries=evidence_dicts if evidence_dicts else [
                {
                    "evidence_id": f"E{i+1}",
                    "modality": sr.modality,
                    "asset_path": sr.path,
                    "timestamp_sec": sr.timestamp_sec,
                    "visual_description": sr.caption or "",
                    "grounding_rationale": f"CLIP score={sr.score:.4f}",
                }
                for i, sr in enumerate(final_fused.results[:self.evidence_top_n])
            ],
        )

        # ===== 构建 Fused Results 摘要 =====
        fused_dicts = []
        for sr in final_fused.results[:10]:
            fused_dicts.append({
                "path": sr.path,
                "score": sr.score,
                "modality": sr.modality,
                "timestamp_sec": sr.timestamp_sec,
                "video_path": sr.video_path,
                "clip_path": sr.clip_path,
            })

        elapsed = (time.perf_counter() - t0) * 1000

        return AgentResponse(
            answer=synthesis.answer,
            citations=synthesis.citations,
            search_trajectory=trajectory_steps,
            evidences=evidence_dicts,
            fused_results=fused_dicts,
            total_rounds=len(trajectory_steps),
            confidence=synthesis.confidence,
            disclaimer=synthesis.disclaimer,
            elapsed_ms=round(elapsed, 1),
            plan={
                "query_type": final_plan.query_type if final_plan else "simple",
                "plan_summary": final_plan.plan_summary if final_plan else "",
                "sub_queries": [
                    {"text": sq.text, "modality": sq.modality, "weight": sq.weight}
                    for sq in (final_plan.sub_queries if final_plan else [])
                ],
                "fusion_strategy": final_plan.fusion_strategy if final_plan else "max",
            },
        )


# ========== 简易测试 ==========
if __name__ == "__main__":
    orchestra = AgentOrchestra(max_rounds=2, enable_llm_rerank=False)
    print("=" * 60)
    print("Agent Orchestra — 端到端测试")
    print("=" * 60)

    test_queries = [
        "一只猫",
        "一只坐在红色沙发上的黑猫",
    ]
    for q in test_queries:
        print(f"\n>>> Query: {q}")
        resp = orchestra.search(q)
        print(f"  Rounds: {resp.total_rounds}")
        print(f"  Confidence: {resp.confidence}")
        print(f"  Elapsed: {resp.elapsed_ms:.0f}ms")
        print(f"  Answer: {resp.answer[:200]}")
        if resp.citations:
            print(f"  Citations: {len(resp.citations)}")
        if resp.disclaimer:
            print(f"  Disclaimer: {resp.disclaimer}")
        print("-" * 40)
