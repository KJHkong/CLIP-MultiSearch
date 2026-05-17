# CLIP-MultiSearch Agent — 完整框架总结

## 一、项目概览

将一个基于 CLIP + FAISS 的单步检索引擎升级为完整的 **Agentic Multimodal Search System**，具备 Planning（规划）→ Routing（路由）→ Searching（检索）→ Reflection（反思）→ Evidence Grounding（证据归因）→ Answer Synthesis（答案合成）的全链路能力。

**技术栈**：CLIP ViT-B/32 (本地) + FAISS + DeepSeek V4 Pro + Qwen3-VL-32B + FastAPI + Gradio

---

## 二、架构总览

```
用户查询(中/英)
     │
     ▼
┌─────────────────────────────────────────────────┐
│ [Phase 1] Planner           DeepSeek V4 Pro      │
│ 分析查询 → 拆解为子查询计划 → 指定模态与权重      │
│ 关键文件: agent_planner.py: plan_query()          │
└────────────────────┬────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────┐
│ [Phase 2] Router + Search     CLIP + FAISS       │
│ 根据子查询模态分发给对应工具(图/视频/以图搜图)     │
│ 关键文件: agent_planner.py: route_and_search()    │
│          search.py: tool_search_*()               │
└────────────────────┬────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────┐
│ [Phase 3] Fusion + Rerank    DeepSeek V4 Pro     │
│ 多路结果去重→加权融合→LLM语义重排序               │
│ 关键文件: agent_planner.py: fuse_results()        │
│          agent_planner.py: llm_rerank()           │
└────────────────────┬────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────┐
│ [Phase 4] Reflection          DeepSeek V4 Pro    │
│ 评判检索质量 → 若不足则改写查询重试(最多3轮)      │
│ 关键文件: agent_reflector.py: reflect()           │
│          agent_reflector.py: SearchMemory         │
└────────────────────┬────────────────────────────┘
                     ▼ (循环至满足或达上限)
┌─────────────────────────────────────────────────┐
│ [Phase 5] Evidence Grounding    Qwen3-VL-32B   │
│ 对Top-N结果做视觉验证→生成帧级描述+相关性理由     │
│ 关键文件: agent_evidence.py: extract_evidence()   │
└────────────────────┬────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────┐
│ [Phase 6] Answer Synthesis     DeepSeek V4 Pro   │
│ 融合搜索轨迹+视觉证据→生成带引用[E1][E2]的答案   │
│ 关键文件: agent_synthesizer.py: synthesize()      │
└────────────────────┬────────────────────────────┘
                     ▼
               AgentResponse
      {answer, citations, evidences,
       trajectory, confidence, elapsed_ms}
```

**主编排器**：`agent_orchestra.py: AgentOrchestra.search()` 负责串联全部 6 个 Phase。

---

## 三、各模块详解

### 3.1 Tool Schema 标准化 (`tools_schema.py`)

**目的**：将所有检索能力抽象为统一的 Tool 接口，供 Agent 规划调用。

| 核心类 | 说明 |
|--------|------|
| `SearchResult` | 单条检索结果：id, modality(image/video_frame), path, score, timestamp_sec, video_path, clip_path, caption |
| `ToolResult` | 一次工具调用的完整返回：tool_name, query, results: List[SearchResult], total_candidates, metadata |
| `ToolDefinition` | 工具注册定义：name, description, modality, func, input_schema |
| `ToolRegistry` | 工具注册中心：register/get/list_tools/call |

三个检索工具：
- `search_image_by_text(query, topk)` — CLIP文本搜图
- `search_video_by_text(query, topk)` — CLIP文本搜视频帧
- `search_image_by_image(image_path, topk)` — CLIP以图搜图

**关键代码**：`tools_schema.py:59-86` (ToolRegistry), `search.py:259-299` (tool包装函数)

---

### 3.2 Planner Agent (`agent_planner.py`)

**目的**：用 DeepSeek V4 Pro 将用户查询拆解为结构化的子查询计划。

**关键数据结构**：
```python
PlanOutput {
    query_type: "simple"|"complex"|"comparative"|"sequential"|"cross_modal"
    sub_queries: [SubQuery{text, modality, weight, rationale}]
    plan_summary, fusion_strategy: "max"|"weighted"|"llm_rerank"
}
```

**关键函数**：
| 函数 | 作用 |
|------|------|
| `plan_query(user_query)` | 分析查询→输出PlanOutput。短查询(≤8词)直接透传避免过度工程化 |
| `route_and_search(plan, tool_registry)` | 按modality分发子查询到对应工具 |
| `fuse_results(tool_results, plan)` | 按path去重+取最高分+降序排列 |
| `llm_rerank(fused, query)` | DeepSeek对整个top-20做语义重排序 |

**关键代码**：`agent_planner.py:34-61` (Planner System Prompt), `agent_planner.py:64-119` (plan_query with fallback), `agent_planner.py:218-249` (fuse_results)

**特点**：LLM调用失败时自动降级为简单透传，无API Key也能正常工作。

---

### 3.3 Reflection Agent (`agent_reflector.py`)

**目的**：让 Agent 自我批判检索质量，不足时自动改写查询重试。

**关键数据结构**：
```python
ReflectionOutput {
    sufficient: bool      # 结果是否充分
    confidence: float     # 0.0-1.0
    missing_info: str     # 缺失什么信息
    next_query: str|null  # 建议的下一轮查询
    reason: str           # 判断理由
}

SearchMemory           # 记录每轮搜索历史，避免冗余查询
```

**工作流程**：
1. 将当前轮所有 ToolResult 的前5条摘要 + 历史搜索记录喂给 DeepSeek V4 Pro
2. LLM判断：结果是否足以回答用户问题？
3. 若 `confidence < 0.6` → 生成 `next_query` → 回到 Planner 重新规划
4. 若 `confidence ≥ 0.6` → 终止循环，进入 Evidence 阶段
5. SearchMemory 防止重复查询（Jaccard > 0.85 视为冗余）

**关键代码**：`agent_reflector.py:90-108` (Reflector System Prompt), `agent_reflector.py:111-183` (reflect with rule-based fallback)

---

### 3.4 Evidence Grounding (`agent_evidence.py`)

**目的**：用 Qwen3-VL-32B 对检索结果做视觉证据定位，返回帧/图级描述和相关性理由（而非单纯的 top-K CLIP 分数）。

**关键数据结构**：
```python
Evidence {
    evidence_id: str          # "E1", "E2", ...
    modality: str             # "image"|"video_frame"
    asset_path: str           # 图片/帧路径
    timestamp_sec: float|null # 视频时间戳
    relevance_score: float    # CLIP分数(若VLM判不相关则×0.3)
    visual_description: str   # VLM对该帧/图的1-2句描述
    grounding_rationale: str  # 为什么此证据与query相关
    bounding_hint: str        # 视觉区域提示("center"/"top-left"/...)
}
```

**工作流程**：
1. 对 FusedResults 的 top-N (默认5) 逐个处理
2. 将图片/帧转 base64 → 连同用户query发给 Qwen3-VL-32B
3. VLM 返回 JSON：`{relevant, visual_description, rationale, bounding_hint}`
4. 若 VLM 判定不相关 → relevance_score × 0.3 降权

**关键代码**：`agent_evidence.py:35-49` (VLM System Prompt), `agent_evidence.py:72-152` (extract_evidence), `agent_evidence.py:175-189` (extract_all_evidences批量处理)

---

### 3.5 Answer Synthesis (`agent_synthesizer.py`)

**目的**：融合搜索轨迹和视觉证据，生成带结构化引用的最终答案。

**关键数据结构**：
```python
SynthesisOutput {
    answer: str              # 最终答案(含[E1][E2]引用标记)
    citations: [Citation]    # {evidence_id, asset_display, note}
    confidence: float        # 答案置信度
    disclaimer: str          # 信息不足时诚实说明
}
```

**关键代码**：`agent_synthesizer.py:42-58` (Synthesizer System Prompt), `agent_synthesizer.py:61-148` (synthesize with fallback)

---

### 3.6 主编排器 (`agent_orchestra.py`)

**目的**：将所有 Phase 串联为完整的 Agent Pipeline。

**核心类**：`AgentOrchestra`

```python
class AgentOrchestra:
    def __init__(self, tool_registry, max_rounds=3, confidence_threshold=0.6,
                 enable_evidence=True, enable_llm_rerank=True, evidence_top_n=5)

    def search(self, user_query) -> AgentResponse:
        # Phase 1-3: Iterative Search Loop (Plan → Route → Search → Fuse → Reflect)
        for round in range(max_rounds):
            plan = plan_query(user_query or reflection.next_query)
            tool_results = route_and_search(plan, self.tool_registry)
            fused = fuse_results(tool_results, plan)
            if enable_llm_rerank: fused = llm_rerank(fused, user_query)
            reflection = reflect(user_query, tool_results, memory)
            if reflection.sufficient and confidence >= threshold: break

        # Phase 4: Evidence Grounding
        evidences = extract_all_evidences(final_fused, user_query)

        # Phase 5: Answer Synthesis
        synthesis = synthesize(user_query, trajectory, evidence_dicts)

        return AgentResponse(answer, citations, trajectory, evidences, ...)
```

**AgentResponse 包含**：answer, citations, search_trajectory, evidences (List[Dict]), fused_results, total_rounds, confidence, disclaimer, elapsed_ms, plan

**关键代码**：`agent_orchestra.py:109-261` (AgentOrchestra完整实现)

---

### 3.7 API 层 (`api_fastapi.py`)

**端点一览**：

| 方法 | 路径 | 功能 |
|------|------|------|
| GET | `/` | 服务信息 |
| GET | `/health` | 健康检查 |
| GET | `/files/{path}` | 图片/视频文件服务 |
| POST | `/search/text` | 文本搜图(单步) |
| POST | `/search/video` | 文本搜视频帧(单步) |
| POST | `/search/image` | 以图搜图(文件上传) |
| POST | `/search/image/json` | 以图搜图(base64) |
| POST | `/search/agent` | **完整Agent Pipeline** |

Agent端点请求/响应模型定义在 `api_fastapi.py:92-145`。

---

### 3.8 评测框架 (`eval_benchmark.py`)

**评测集**：`eval/queries_complex.json` — 70条中文查询，覆盖5个难度等级：
| 等级 | 描述 | 数量 |
|------|------|------|
| L1_simple | 简单描述/单模态 | 10 |
| L2_multi_attr | 多属性约束 | 15 |
| L3_cross_modal | 跨模态(图+视频) | 15 |
| L4_temporal | 时序/顺序 | 15 |
| L5_multi_hop | 多跳推理 | 15 |

**消融实验配置**：
| 配置 | 说明 |
|------|------|
| `baseline` | 单次CLIP检索(expand_n=1, 无LLM) |
| `multi_prompt` | 现有多Prompt LLM改写+Max融合(expand_n=6) |
| `agent_full` | 完整Agent: Planner + Reflection + Evidence + Synthesis |

**评测指标**：Recall@K (K=1,5,10)、nDCG@K、平均检索轮数、端到端延迟(ms)、置信度

**使用方式**：
```bash
python -m src.eval_benchmark                  # 全量
python -m src.eval_benchmark --mode quick     # 前10条测试
python -m src.eval_benchmark --config baseline # 仅baseline
```

---

### 3.9 Gradio UI (`ui_gradio.py`)

**四个 Tab**：
| Tab | 功能 | 对应函数 |
|-----|------|----------|
| 文本检索 | 文本搜图+多prompt融合 | `run()` → `search_with_fusion()` |
| 以图搜图 | 上传图片搜相似图 | `run_image()` → `search_by_image()` |
| 捕捉视频关键帧 | 文本搜视频帧+播放片段 | `run_video()` → `search_video_clips()` |
| Agent Search | 完整Agent Pipeline可视化 | `run_agent()` → `AgentOrchestra.search()` |

**Agent Search Tab 展示内容**：
- Plan（查询规划+子查询列表）
- Search Trajectory（每轮搜索结果+Reflection判断）
- Evidence（视觉证据+描述+相关性理由）
- Citations（引用列表）
- Final Answer（含[E1][E2]引用的最终答案）
- Top Results Gallery（检索结果预览）
- Status（轮数/置信度/延迟）

---

## 四、数据流全程追踪

以查询 _"一只坐在红色沙发上的黑猫"_ 为例：

```
1. UI/api → query="一只坐在红色沙发上的黑猫"
2. AgentOrchestra.search(query)
3. Phase1: plan_query() → PlanOutput{
     query_type="complex",
     sub_queries=[SubQuery("a black cat sitting on a red sofa", modality="both", weight=1.0)],
     plan_summary="多属性约束查询，直接检索",
     fusion_strategy="max"
   }
4. Phase2: route_and_search() → [
     ToolResult("search_image_by_text", results=[SearchResult×20]),
     ToolResult("search_video_by_text", results=[SearchResult×5]),
   ]
5. Phase3: fuse_results() → FusedResults(results=[SearchResult×25去重排序])
   llm_rerank() → FusedResults(results=[SearchResult×10语义重排])
6. Phase4: reflect() → ReflectionOutput{
     sufficient=True, confidence=0.75,
     reason="top结果命中红色沙发上的黑猫"
   }
   → 满足阈值，跳出循环，共1轮
7. Phase5: extract_all_evidences(top5) → [
     Evidence(E1, "A black cat sitting on a red velvet sofa", score=0.35),
     Evidence(E2, "A dark-colored cat on red furniture", score=0.28),
     ...
   ]
8. Phase6: synthesize(trajectory, evidences) → SynthesisOutput{
     answer="根据检索结果，找到了一只坐在红色沙发上的黑猫[E1]...",
     citations=[Citation(E1, "images/cat_sofa.jpg", "A black cat..."), ...],
     confidence=0.85
   }
9. 返回 AgentResponse 给调用方
```

---

## 五、关键设计决策

1. **降级容错**：所有LLM/VLM依赖模块都有 rule-based fallback，API Key缺失或网络故障时系统仍可运行
2. **工具抽象**：新增模态只需注册 ToolDefinition，编排层无需改动
3. **Memory防冗余**：SearchMemory 记录每轮搜索，Jaccard相似度 > 0.85 阻止重复
4. **置信度阈值 0.6**：Reflection循环在 confidence ≥ 0.6 或 max_rounds 耗尽时终止
5. **VLM降权**：若 Qwen-VL 判定某结果不相关 → 分数 × 0.3，而非直接丢弃
