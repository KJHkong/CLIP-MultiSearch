"""
Evidence Grounding Agent：用 Qwen2.5-VL 对检索结果做视觉证据定位，
返回 frame/page 级证据引用（描述 + 相关性理由）。
参考 Ground-R1 的"先 grounding 再回答"范式、MEG-RAG 的证据质量度量思路。
"""
import base64
import json
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path
from openai import OpenAI

from src.config import VLM_API_BASE, VLM_API_KEY, VLM_MODEL
from src.tools_schema import SearchResult
from src.agent_planner import FusedResults


@dataclass
class Evidence:
    """单条证据。"""
    evidence_id: str                               # E1, E2, ...
    modality: str                                  # "image" | "video_frame"
    asset_path: str                                # 图片/帧路径
    timestamp_sec: Optional[float] = None
    video_path: Optional[str] = None
    clip_path: Optional[str] = None
    relevance_score: float = 0.0                   # CLIP 分数
    visual_description: str = ""                   # VLM 对该帧/图的简述
    grounding_rationale: str = ""                  # 为什么此证据与 query 相关
    bounding_hint: str = ""                        # 视觉区域提示（如"画面中央偏左"）


# ---------- VLM System Prompt ----------

VLM_EVIDENCE_PROMPT = """You are a visual evidence evaluator. Given an image (or video frame) and a user's search query, determine:

1. Is this image/frame relevant to the query?
2. If yes, describe the visual content in 1-2 short sentences (in English).
3. Explain in 1 sentence why this visual evidence supports the query.

Output a JSON object:
{
  "relevant": true or false,
  "visual_description": "short description of what you see that relates to the query",
  "rationale": "why this image relates to the query",
  "bounding_hint": "where in the image the key content is located (e.g. 'center', 'top-left', 'bottom-right', 'full frame')"
}

Output ONLY the JSON object, no extra text."""


def _image_to_base64(image_path: str) -> str:
    """将本地图片/帧转为 base64 data URL。"""
    path = Path(image_path)
    if not path.exists():
        # 尝试项目根解析
        proj_root = Path(__file__).resolve().parents[1]
        path = proj_root / image_path
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    suffix = path.suffix.lower()
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp", "bmp": "bmp"}.get(
        suffix.lstrip("."), "jpeg"
    )

    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/{mime};base64,{b64}"


def extract_evidence(
    search_result: SearchResult,
    user_query: str,
    evidence_id: str,
) -> Evidence:
    """
    用 Qwen2.5-VL 对单条检索结果做证据定位。

    参数：
        search_result: 一条检索结果
        user_query: 用户原始查询
        evidence_id: 证据编号（如 "E1"）

    返回：
        Evidence: 带视觉描述和相关性的证据对象
    """
    evidence = Evidence(
        evidence_id=evidence_id,
        modality=search_result.modality,
        asset_path=search_result.path,
        timestamp_sec=search_result.timestamp_sec,
        video_path=search_result.video_path,
        clip_path=search_result.clip_path,
        relevance_score=search_result.score,
    )

    if not VLM_API_KEY:
        evidence.visual_description = "(VLM 未配置)"
        evidence.grounding_rationale = "CLIP 相似度匹配"
        evidence.bounding_hint = "full frame"
        return evidence

    # 如果是视频帧但没有 clip_path，尝试直接加载帧
    try:
        img_b64 = _image_to_base64(search_result.path)
    except FileNotFoundError as e:
        evidence.visual_description = f"(图片不可访问: {e})"
        evidence.grounding_rationale = "文件缺失"
        return evidence

    try:
        client = OpenAI(base_url=VLM_API_BASE, api_key=VLM_API_KEY)
        response = client.chat.completions.create(
            model=VLM_MODEL,
            messages=[
                {"role": "system", "content": VLM_EVIDENCE_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": img_b64},
                        },
                        {
                            "type": "text",
                            "text": f"User query: {user_query}\n\nEvaluate this image's relevance and output JSON:",
                        },
                    ],
                },
            ],
            max_tokens=300,
            temperature=0.1,
        )
        raw = (response.choices[0].message.content or "").strip()
        data = _parse_vlm_json(raw)

        if data.get("relevant", True):
            evidence.visual_description = data.get("visual_description", "")
            evidence.grounding_rationale = data.get("rationale", "")
            evidence.bounding_hint = data.get("bounding_hint", "full frame")
        else:
            evidence.visual_description = "(VLM 判定不相关)"
            evidence.grounding_rationale = data.get("rationale", "VLM 判定与查询不相关")
            evidence.relevance_score *= 0.3  # 降权

    except Exception as e:
        print(f"[Evidence] VLM 调用失败 ({evidence_id}:{search_result.path[:40]}): {e}")
        evidence.visual_description = f"(VLM 错误: {str(e)[:60]})"
        evidence.grounding_rationale = "VLM 调用失败，仅依赖 CLIP 分数"

    return evidence


def _parse_vlm_json(raw: str) -> dict:
    """解析 VLM 返回的 JSON。"""
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
    return {}


def extract_all_evidences(
    fused: FusedResults,
    user_query: str,
    top_n: int = 5,
) -> List[Evidence]:
    """
    对融合后的 top-N 结果批量做证据定位。
    对每张图片/帧调用 Qwen2.5-VL 判定相关性并生成描述。
    """
    evidences = []
    for i, sr in enumerate(fused.results[:top_n]):
        eid = f"E{i + 1}"
        ev = extract_evidence(sr, user_query, eid)
        evidences.append(ev)
    return evidences


# ========== 简易测试 ==========
if __name__ == "__main__":
    sr = SearchResult(
        id=0, modality="image", path="data/images/test.jpg", score=0.35,
    )
    ev = extract_evidence(sr, "a cat on a sofa", "E1")
    print(f"  ID: {ev.evidence_id}")
    print(f"  Score: {ev.relevance_score}")
    print(f"  Visual: {ev.visual_description}")
    print(f"  Rationale: {ev.grounding_rationale}")
    print(f"  Bounding: {ev.bounding_hint}")
