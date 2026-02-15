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

# 硅基流动：国内用 .cn，国际用 .com；Key 无效(401) 时可试另一个
# 方式1：终端 set SILICONFLOW_API_KEY=你的key  方式2：.env 里写 SILICONFLOW_API_KEY=你的key
LLM_API_BASE = os.environ.get("SILICONFLOW_API_BASE", "https://api.siliconflow.cn/v1")
LLM_API_KEY = os.environ.get("SILICONFLOW_API_KEY", "")
# 硅基流动上的 DeepSeek 模型名，可按控制台实际名称修改
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-ai/DeepSeek-V3")
