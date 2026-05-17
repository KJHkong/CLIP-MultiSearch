# Agentic Multimodal Search — System Architecture

## Overview

CLIP-MultiSearch Agent extends single-step CLIP retrieval into a full **Agentic Multimodal Search Pipeline**: Plan → Route → Search → Reflect → Evidence → Synthesize. Inspired by ViDoRAG (2025) and VSA frameworks.

## Architecture Diagram

```
User Query (Chinese/English)
       │
       ▼
┌──────────────────────────────────────────────┐
│  Phase 1: Planner (agent_planner.py)         │
│  DeepSeek V4 Pro decomposes query into       │
│  structured sub-queries with modalities.     │
│  Output: PlanOutput {query_type, sub_queries}│
└──────────────────┬───────────────────────────┘
                   ▼
┌──────────────────────────────────────────────┐
│  Phase 2: Router (agent_planner.py)          │
│  route_and_search() dispatches sub-queries   │
│  to correct tools (image/video/both).        │
│  Output: List[ToolResult]                    │
└──────────────────┬───────────────────────────┘
                   ▼
┌──────────────────────────────────────────────┐
│  Phase 3: Fuse + Rerank (agent_planner.py)   │
│  fuse_results() deduplicates by path,        │
│  llm_rerank() re-orders semantically.        │
│  Output: FusedResults                        │
└──────────────────┬───────────────────────────┘
                   ▼
┌──────────────────────────────────────────────┐
│  Phase 4: Reflector (agent_reflector.py)     │
│  DeepSeek V4 Pro evaluates result quality.   │
│  If confidence < 0.6 → re-plan with new query│
│  If sufficient → proceed to evidence phase.  │
│  SearchMemory prevents redundant queries.    │
└──────────────────┬───────────────────────────┘
                   ▼ (loop up to max_rounds)
┌──────────────────────────────────────────────┐
│  Phase 5: Evidence Grounding (agent_evidence) │
│  Qwen3-VL-32B examines top-N results       │
│  visually. Produces descriptions, bounding   │
│  hints, and relevance rationales.            │
│  Output: List[Evidence]                      │
└──────────────────┬───────────────────────────┘
                   ▼
┌──────────────────────────────────────────────┐
│  Phase 6: Synthesizer (agent_synthesizer.py) │
│  DeepSeek V4 Pro generates final answer      │
│  with structured citations [E1][E2]...       │
│  Output: SynthesisOutput {answer, citations} │
└──────────────────┬───────────────────────────┘
                   ▼
            AgentResponse
```

## Module Map

| File | Role | LLM |
|------|------|-----|
| `tools_schema.py` | Unified Tool interface (ToolResult, ToolRegistry, SearchResult) | — |
| `agent_planner.py` | Query decomposition + modal routing + result fusion | DeepSeek V4 Pro |
| `agent_reflector.py` | Self-critique loop + search memory | DeepSeek V4 Pro |
| `agent_evidence.py` | Visual evidence grounding with frame-level citations | Qwen3-VL-32B |
| `agent_synthesizer.py` | Answer synthesis with structured citations | DeepSeek V4 Pro |
| `agent_orchestra.py` | Main orchestrator wiring all phases | — |
| `search.py` | CLIP + FAISS core engine + Tool-wrapped functions | CLIP (local) |

## API Resources

| API | Provider | Used For |
|-----|----------|----------|
| DeepSeek V4 Pro | api.deepseek.com | Planner, Reflector, Synthesizer, Rerank, Query Rewrite |
| Qwen3-VL-32B | api.siliconflow.cn | Evidence Grounding (visual verification) |
| CLIP ViT-B/32 | Local | Core similarity search (text↔image, image↔image, text↔video frame) |

## Data Flow

1. **Search** produces `ToolResult` (standardized)
2. **Fuse** deduplicates by path, keeps highest CLIP score per asset
3. **Rerank** uses LLM to semantically re-rank top candidates
4. **Evidence** attaches VLM-generated descriptions to each result
5. **Synthesize** consumes trajectory + evidence → final answer with inline citations

## Key Design Decisions

- **Graceful degradation**: All LLM-dependent modules fall back to rule-based logic when API keys are unavailable
- **Tool abstraction**: Search functions are wrapped as `ToolResult` so new modalities (e.g., audio, PDF) can be added without changing the orchestration layer
- **Fusion strategy**: Max-pooling across expanded prompts, with optional LLM re-rank for complex queries
- **Confidence threshold (0.6)**: Reflection loop continues until confidence ≥ 0.6 or max_rounds exhausted

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /search/text` | Text-to-image/video search (single step) |
| `POST /search/image` | Image-to-image search |
| `POST /search/video` | Text-to-video keyframe search |
| `POST /search/agent` | **Full Agent pipeline** (Plan→Search→Reflect→Evidence→Synthesize) |

## Evaluation

See `eval/` directory:
- `queries_complex.json`: 80 curated queries across 5 difficulty levels
- `eval_benchmark.py`: Ablation study framework comparing baseline → multi_prompt → agent_full
- Metrics: Recall@K, nDCG@K, avg rounds, latency, confidence
