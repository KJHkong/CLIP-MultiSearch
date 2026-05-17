"""
统一 Tool Schema：将所有检索能力抽象为结构化 Tool 接口，供 Agent 规划调用。
"""
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Callable
from enum import Enum
import time
import uuid


class Modality(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    BOTH = "both"


@dataclass
class SearchResult:
    """单条检索结果。"""
    id: int
    modality: str                    # "image" | "video_frame"
    path: str                        # 资源路径
    score: float                     # CLIP 相似度
    timestamp_sec: Optional[float] = None
    video_path: Optional[str] = None
    clip_path: Optional[str] = None
    caption: Optional[str] = None    # 可选描述


@dataclass
class ToolResult:
    """一次工具调用的完整返回。"""
    tool_name: str
    query: str
    results: List[SearchResult] = field(default_factory=list)
    total_candidates: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def topk_paths(self, k: int = 5) -> List[str]:
        return [r.path for r in self.results[:k]]

    def summary(self) -> str:
        if not self.results:
            return f"[{self.tool_name}] 无结果"
        top = self.results[0]
        return f"[{self.tool_name}] top1={top.path} score={top.score:.4f} total={len(self.results)}"


@dataclass
class ToolDefinition:
    """工具注册定义。"""
    name: str
    description: str
    modality: Modality
    func: Callable
    input_schema: Dict[str, Any] = field(default_factory=dict)


class ToolRegistry:
    """工具注册中心：管理所有可用检索工具。"""

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}

    def register(self, td: ToolDefinition):
        self._tools[td.name] = td

    def get(self, name: str) -> Optional[ToolDefinition]:
        return self._tools.get(name)

    def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": td.name,
                "description": td.description,
                "modality": td.modality.value,
                "input_schema": td.input_schema,
            }
            for td in self._tools.values()
        ]

    def call(self, name: str, **kwargs) -> ToolResult:
        td = self._tools.get(name)
        if td is None:
            raise KeyError(f"Tool '{name}' not registered. Available: {list(self._tools)}")
        return td.func(**kwargs)


# ---------- 将现有 search 函数包装为 ToolResult ----------

def wrap_as_tool_result(
    results: List[Dict],
    tool_name: str,
    query: str,
    metadata: Optional[Dict] = None,
) -> ToolResult:
    """将 search.py 的 dict 输出包装成 ToolResult。"""
    sr_list = []
    for r in results:
        sr_list.append(SearchResult(
            id=r.get("id", -1),
            modality=r.get("type", "image"),
            path=r.get("path", ""),
            score=round(float(r.get("score", 0)), 4),
            timestamp_sec=r.get("timestamp_sec"),
            video_path=r.get("video_path"),
            clip_path=r.get("clip_path"),
            caption=r.get("caption"),
        ))
    return ToolResult(
        tool_name=tool_name,
        query=query,
        results=sr_list,
        total_candidates=len(sr_list),
        metadata=metadata or {},
    )
