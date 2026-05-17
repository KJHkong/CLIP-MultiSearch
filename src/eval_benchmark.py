"""
评测框架 + 消融实验：量化每个 Agent 模块的贡献，产出可写入简历的消融实验结果。

支持配置：
  - baseline: 单次 CLIP 检索（无多 prompt 融合）
  - +multi_prompt: 现有多 Prompt 融合（当前系统）
  - +planner: Planner + 模态路由
  - +reflection: Planner + Reflection Loop
  - +evidence: 完整全链路（含证据归因 + Answer Synthesis）

指标：
  - Recall@K (K=1,5,10)
  - nDCG@K
  - 平均检索轮数
  - 端到端延迟 (ms)
  - Answer Confidence (LLM 自评)

使用方式：
  python -m src.eval_benchmark                  # 运行所有配置
  python -m src.eval_benchmark --mode quick     # 快速测试（仅前10条query）
  python -m src.eval_benchmark --config baseline # 仅运行 baseline
"""
import json
import time
import sys
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.agent_orchestra import AgentOrchestra, AgentResponse
from src.search import search_with_fusion


# ---------- 评测指标 ----------

@dataclass
class EvalMetrics:
    """单条 query 的评测指标。"""
    query_id: str
    query: str
    config: str
    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    ndcg_at_5: float = 0.0
    ndcg_at_10: float = 0.0
    num_results: int = 0
    num_rounds: int = 1
    elapsed_ms: float = 0.0
    confidence: float = 0.0
    answer: str = ""
    error: str = ""


@dataclass
class AblationReport:
    """消融实验报告。"""
    config_name: str
    description: str
    queries: int
    avg_recall_1: float = 0.0
    avg_recall_5: float = 0.0
    avg_recall_10: float = 0.0
    avg_ndcg_5: float = 0.0
    avg_ndcg_10: float = 0.0
    avg_rounds: float = 0.0
    avg_latency_ms: float = 0.0
    avg_confidence: float = 0.0
    errors: int = 0
    per_query: List[EvalMetrics] = field(default_factory=list)


# ---------- 评测核心 ----------

def load_queries(subset: Optional[int] = None) -> List[Dict]:
    """加载评测 query 集。"""
    queries_file = _root / "eval" / "queries_complex.json"
    if not queries_file.exists():
        print(f"[Eval] Query file not found: {queries_file}")
        return []
    with open(queries_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    queries = data.get("queries", [])
    if subset:
        queries = queries[:subset]
    return queries


def _dcg(scores: List[float], k: int) -> float:
    """Discounted Cumulative Gain。"""
    import math
    dcg = 0.0
    for i, s in enumerate(scores[:k]):
        dcg += (2 ** s - 1) / math.log2(i + 2)
    return dcg


def compute_metrics(
    query_id: str,
    query: str,
    config_name: str,
    results: List[Dict],
    num_rounds: int,
    elapsed_ms: float,
    confidence: float,
    answer: str,
    error: str = "",
) -> EvalMetrics:
    """从检索结果计算 Recall@K 和 nDCG@K（以 CLIP score 为相关性得分）。"""
    m = EvalMetrics(
        query_id=query_id, query=query, config=config_name,
        num_results=len(results), num_rounds=num_rounds,
        elapsed_ms=elapsed_ms, confidence=confidence, answer=answer, error=error,
    )

    if not results:
        return m

    scores = [r.get("score", 0.0) for r in results]

    # Recall@K: 以 top-K 中 score > 0.25 的比例作为弱监督 recall
    threshold = 0.25
    m.recall_at_1 = 1.0 if scores[0] >= threshold else 0.0
    m.recall_at_5 = sum(1 for s in scores[:5] if s >= threshold) / min(5, len(scores))
    m.recall_at_10 = sum(1 for s in scores[:10] if s >= threshold) / min(10, len(scores))

    # nDCG: 用 score 作为 relevance
    ideal = sorted(scores, reverse=True)
    m.ndcg_at_5 = _dcg(scores, 5) / _dcg(ideal, 5) if _dcg(ideal, 5) > 0 else 0.0
    m.ndcg_at_10 = _dcg(scores, 10) / _dcg(ideal, 10) if _dcg(ideal, 10) > 0 else 0.0

    return m


def run_baseline(queries: List[Dict]) -> AblationReport:
    """Baseline: 单次 CLIP 检索（无多 prompt 融合）。"""
    report = AblationReport(
        config_name="baseline",
        description="单次 CLIP 检索，expand_n=1",
    )
    for q in queries:
        t0 = time.perf_counter()
        try:
            results, _, _ = search_with_fusion(
                user_query=q["query_en"], topk=10, expand_n=1, use_llm=False,
            )
            elapsed = (time.perf_counter() - t0) * 1000
            m = compute_metrics(
                q["id"], q["query_cn"], "baseline", results,
                num_rounds=1, elapsed_ms=elapsed, confidence=0.0, answer="",
            )
        except Exception as e:
            m = EvalMetrics(query_id=q["id"], query=q["query_cn"], config="baseline", error=str(e))
        report.per_query.append(m)

    report.queries = len(report.per_query)
    report.errors = sum(1 for m in report.per_query if m.error)
    _aggregate(report)
    return report


def run_multi_prompt(queries: List[Dict]) -> AblationReport:
    """+multi_prompt: 现有多 Prompt 融合（当前系统）。"""
    report = AblationReport(
        config_name="multi_prompt",
        description="多 Prompt LLM 改写 + Max 融合，expand_n=6",
    )
    for q in queries:
        t0 = time.perf_counter()
        try:
            results, _, _ = search_with_fusion(
                user_query=q["query_en"], topk=10, expand_n=6, use_llm=True,
            )
            elapsed = (time.perf_counter() - t0) * 1000
            m = compute_metrics(
                q["id"], q["query_cn"], "multi_prompt", results,
                num_rounds=1, elapsed_ms=elapsed, confidence=0.0, answer="",
            )
        except Exception as e:
            m = EvalMetrics(query_id=q["id"], query=q["query_cn"], config="multi_prompt", error=str(e))
        report.per_query.append(m)

    report.queries = len(report.per_query)
    report.errors = sum(1 for m in report.per_query if m.error)
    _aggregate(report)
    return report


def run_agent_full(
    queries: List[Dict],
    max_rounds: int = 3,
    enable_evidence: bool = True,
    enable_rerank: bool = True,
) -> AblationReport:
    """完整 Agent Pipeline: Planner + Reflection + Evidence + Synthesis。"""
    config_label = "agent_full"
    desc = "完整 Agent: Planner + Reflection + Evidence + Synthesis"
    if not enable_evidence:
        config_label = "agent_no_evidence"
        desc = "Agent: Planner + Reflection (no Evidence Grounding)"
    if not enable_rerank:
        config_label += "_no_rerank"

    report = AblationReport(config_name=config_label, description=desc)

    try:
        orchestra = AgentOrchestra(
            max_rounds=max_rounds,
            enable_evidence=enable_evidence,
            enable_llm_rerank=enable_rerank,
        )
    except Exception as e:
        report.errors = len(queries)
        report.queries = len(queries)
        report.per_query = [
            EvalMetrics(query_id=q["id"], query=q["query_cn"], config=config_label, error=str(e))
            for q in queries
        ]
        return report

    for q in queries:
        try:
            resp = orchestra.search(q["query_en"])
            results = resp.fused_results
            m = compute_metrics(
                q["id"], q["query_cn"], config_label, results,
                num_rounds=resp.total_rounds, elapsed_ms=resp.elapsed_ms,
                confidence=resp.confidence, answer=resp.answer[:200],
            )
        except Exception as e:
            m = EvalMetrics(query_id=q["id"], query=q["query_cn"], config=config_label, error=str(e))
        report.per_query.append(m)

    report.queries = len(report.per_query)
    report.errors = sum(1 for m in report.per_query if m.error)
    _aggregate(report)
    return report


def _aggregate(report: AblationReport):
    """汇总指标。"""
    valid = [m for m in report.per_query if not m.error]
    if not valid:
        return
    n = len(valid)
    report.avg_recall_1 = sum(m.recall_at_1 for m in valid) / n
    report.avg_recall_5 = sum(m.recall_at_5 for m in valid) / n
    report.avg_recall_10 = sum(m.recall_at_10 for m in valid) / n
    report.avg_ndcg_5 = sum(m.ndcg_at_5 for m in valid) / n
    report.avg_ndcg_10 = sum(m.ndcg_at_10 for m in valid) / n
    report.avg_rounds = sum(m.num_rounds for m in valid) / n
    report.avg_latency_ms = sum(m.elapsed_ms for m in valid) / n
    report.avg_confidence = sum(m.confidence for m in valid) / n


# ---------- 报告生成 ----------

def print_report(report: AblationReport):
    """打印单份报告。"""
    print(f"\n{'='*60}")
    print(f"Config: {report.config_name}")
    print(f"  Description: {report.description}")
    print(f"  Queries: {report.queries} | Errors: {report.errors}")
    print(f"  ──────────────────────────────────────")
    print(f"  Recall@1:    {report.avg_recall_1:.4f}")
    print(f"  Recall@5:    {report.avg_recall_5:.4f}")
    print(f"  Recall@10:   {report.avg_recall_10:.4f}")
    print(f"  nDCG@5:      {report.avg_ndcg_5:.4f}")
    print(f"  nDCG@10:     {report.avg_ndcg_10:.4f}")
    print(f"  Avg Rounds:  {report.avg_rounds:.2f}")
    print(f"  Avg Latency: {report.avg_latency_ms:.0f}ms")
    print(f"  Avg Confidence: {report.avg_confidence:.4f}")


def generate_ablation_table(reports: List[AblationReport]) -> str:
    """生成 Markdown 消融表格。"""
    lines = [
        "## 消融实验结果",
        "",
        "| Config | R@1 | R@5 | R@10 | nDCG@5 | nDCG@10 | Rounds | Latency(ms) | Confidence | Errors |",
        "|--------|-----|-----|------|--------|---------|--------|-------------|------------|--------|",
    ]
    for r in reports:
        lines.append(
            f"| {r.config_name} | {r.avg_recall_1:.4f} | {r.avg_recall_5:.4f} | "
            f"{r.avg_recall_10:.4f} | {r.avg_ndcg_5:.4f} | {r.avg_ndcg_10:.4f} | "
            f"{r.avg_rounds:.1f} | {r.avg_latency_ms:.0f} | {r.avg_confidence:.4f} | "
            f"{r.errors} |"
        )
    return "\n".join(lines)


def identify_success_stories(reports: List[AblationReport]) -> List[str]:
    """识别 baseline 失败但 agent_full 成功的典型案例。"""
    baseline = next((r for r in reports if r.config_name == "baseline"), None)
    agent = next((r for r in reports if r.config_name == "agent_full"), None)
    if not baseline or not agent:
        return []

    stories = []
    bl_map = {m.query_id: m for m in baseline.per_query}
    ag_map = {m.query_id: m for m in agent.per_query}

    for qid, bl_m in bl_map.items():
        ag_m = ag_map.get(qid)
        if not ag_m or ag_m.error:
            continue
        # baseline 低 recall 但 agent 高 recall 的 case
        if bl_m.recall_at_5 < 0.3 and ag_m.recall_at_5 > 0.5:
            stories.append(
                f"- **{qid}**: `{bl_m.query[:60]}` "
                f"R@5: {bl_m.recall_at_5:.2f} → {ag_m.recall_at_5:.2f} "
                f"({ag_m.num_rounds} rounds)"
            )
    return stories


# ---------- 主入口 ----------

def main():
    parser = argparse.ArgumentParser(description="Agentic Multimodal Search — 评测+消融")
    parser.add_argument("--mode", choices=["full", "quick"], default="full",
                        help="full=全量80条, quick=前10条快速测试")
    parser.add_argument("--config", choices=["baseline", "multi_prompt", "agent", "all"],
                        default="all", help="运行哪个配置")
    parser.add_argument("--output", type=str, default=None,
                        help="输出 JSON 报告路径 (默认 eval/results/)")
    args = parser.parse_args()

    subset = 10 if args.mode == "quick" else None
    queries = load_queries(subset)
    if not queries:
        print("[Eval] 无查询加载，退出。")
        return

    print(f"[Eval] 加载 {len(queries)} 条查询 (mode={args.mode})")
    reports: List[AblationReport] = []

    if args.config in ("baseline", "all"):
        print("\n>>> Running: baseline (单次CLIP)...")
        reports.append(run_baseline(queries))
        print_report(reports[-1])

    if args.config in ("multi_prompt", "all"):
        print("\n>>> Running: multi_prompt (多Prompt融合)...")
        reports.append(run_multi_prompt(queries))
        print_report(reports[-1])

    if args.config in ("agent", "all"):
        print("\n>>> Running: agent_full (完整Agent Pipeline)...")
        reports.append(run_agent_full(queries, max_rounds=3, enable_evidence=True, enable_rerank=True))
        print_report(reports[-1])

    # 生成消融汇总
    if len(reports) >= 2:
        print("\n" + "=" * 60)
        print(generate_ablation_table(reports))

        stories = identify_success_stories(reports)
        if stories:
            print("\n### 典型案例 (baseline → agent 提升)")
            for s in stories:
                print(s)

    # 保存结果
    out_dir = _root / "eval" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = args.output or str(out_dir / f"ablation_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": ts,
            "mode": args.mode,
            "num_queries": len(queries),
            "reports": [
                {
                    "config_name": r.config_name,
                    "description": r.description,
                    "avg_recall_1": r.avg_recall_1,
                    "avg_recall_5": r.avg_recall_5,
                    "avg_recall_10": r.avg_recall_10,
                    "avg_ndcg_5": r.avg_ndcg_5,
                    "avg_ndcg_10": r.avg_ndcg_10,
                    "avg_rounds": r.avg_rounds,
                    "avg_latency_ms": r.avg_latency_ms,
                    "avg_confidence": r.avg_confidence,
                    "errors": r.errors,
                    "per_query": [
                        {
                            "query_id": m.query_id,
                            "recall_1": m.recall_at_1,
                            "recall_5": m.recall_at_5,
                            "recall_10": m.recall_at_10,
                            "ndcg_5": m.ndcg_at_5,
                            "ndcg_10": m.ndcg_at_10,
                            "rounds": m.num_rounds,
                            "elapsed_ms": m.elapsed_ms,
                            "confidence": m.confidence,
                            "error": m.error,
                        }
                        for m in r.per_query
                    ],
                }
                for r in reports
            ],
        }, f, ensure_ascii=False, indent=2)
    print(f"\n[Eval] 报告已保存: {out_path}")


if __name__ == "__main__":
    main()
