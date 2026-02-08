# 🔍 CLIP-MultiSearch

**多模态智能搜索系统 | 文本搜图 · 以图搜图 · 视频关键帧搜索**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyPI license](https://img.shields.io/pypi/l/ansicolortags.svg)](LICENSE)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/CLIP-MultiSearch)

> 基于OpenAI CLIP的多模态搜索系统，支持文本、图像、视频的智能语义搜索

## ✨ 核心功能

| 功能 | 描述 | 状态 |
|------|------|------|
| 📝 **文本搜索图像** | 使用自然语言描述搜索图片库 | ✅ 已实现 |
| 🖼️ **以图搜图** | 上传图片查找相似图片 | 🔄 开发中 |
| 🎬 **视频关键帧搜索** | 提取视频关键帧并支持搜索 | 🔄 开发中 |
| 🔄 **查询智能扩展** | 自动扩展查询提升召回率 | ✅ 已实现 |
| 🌐 **Web交互界面** | 基于Gradio的直观界面 | ✅ 已实现 |

🖼️ Interface Preview
https://github.com/KJHkong/CLIP-MultiSearch/issues/1#issue-3912913840

Search interface for query "a girl" showing image results with similarity scores

## 🚀 快速开始

```bash
# 克隆仓库
git clone https://github.com/yourusername/CLIP-MultiSearch.git
cd CLIP-MultiSearch

# 安装依赖
pip install -r requirements.txt

# 准备数据
mkdir -p data/images
# 将你的图片放入 data/images/

# 构建索引
python src/build_index.py

# 启动Web界面
python src/web/gradio_ui.py
