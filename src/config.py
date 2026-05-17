# LLM 改写相关配置（硅基流动 DeepSeek API，OpenAI 兼容）
import os
from pathlib import Path

# 从 .env 加载 API Key：先找 config 所在目录（src/），再找项目根目录
def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip("'\"").strip()
            if k and v and k not in os.environ:
                os.environ[k] = v

_config_dir = Path(__file__).resolve().parent
_root_dir = _config_dir.parent
for _env_file in (_config_dir / ".env", _root_dir / ".env"):
    if _env_file.exists():
        _load_dotenv(_env_file)
        break

# ========== 文本推理 — DeepSeek 官方 API ==========
# Planner / Reflection / Synthesis / Rerank / Query Rewrite
LLM_API_BASE = os.environ.get("LLM_API_BASE", "https://api.deepseek.com")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-chat")

# ========== 视觉模型 — 硅基流动 Qwen-VL ==========
# Evidence Grounding (需多模态能力)
VLM_API_BASE = os.environ.get("VLM_API_BASE", "https://api.siliconflow.cn/v1")
VLM_API_KEY = os.environ.get("VLM_API_KEY", "")
VLM_MODEL = os.environ.get("VLM_MODEL", "Qwen/Qwen3-VL-32B-Instruct")
