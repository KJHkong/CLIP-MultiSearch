# 导入Gradio库：用于快速构建机器学习Web界面
import sys
from pathlib import Path

# 保证从项目根或 src 目录运行都能正确导入
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import gradio as gr
from src.search import search_with_fusion

def run(query: str, topk: int, expand_n: int, use_llm: bool):
    """
    处理用户搜索请求的核心函数
    
    参数：
    - query: 用户输入的搜索文本
    - topk: 返回的结果数量
    - expand_n: 扩展的查询数量
    - use_llm: 是否用 LLM 改写查询（中英文均可）
    
    返回：
    - 图片画廊数据（用于显示图片）
    - 调试信息文本
    """
    # 1. 调用搜索函数，获取结果、调试信息和扩展方式
    results, prompt_scores, expand_source = search_with_fusion(
        user_query=query,
        topk=topk,
        expand_n=expand_n,
        use_llm=use_llm,
    )

    # 2. 准备图片展示数据
    gallery = []  # 存储要显示的图片列表，格式：[(图片路径, 说明文字), ...]
    caption_lines = []  # 这个变量定义但未使用，是预留的

    # 3. 遍历所有搜索结果
    for r in results:
        path = r["path"]  # 图片文件路径
        score = r["score"]  # 相似度分数

        # 判断结果类型：图片或视频（预留扩展）
        if r.get("type") == "image":
            gallery.append((path, f"{score:.4f}  {path}"))
        else:
            # 视频类型（预留功能）：显示视频帧，包含时间戳信息
            gallery.append((path, f"{score:.4f}  {r.get('video_path','')} @ {r.get('timestamp_sec','')}"))

    # 4. 准备调试信息：先显示本次用的扩展方式，再展示每个 prompt 的 top1 分数
    debug = f"[扩展方式] {expand_source}\n\n" + "\n".join([f"{s:.4f}\t{p}" for p, s in prompt_scores])

    # 5. 返回结果给Gradio显示
    return gallery, debug

# 6. 创建Gradio界面
with gr.Blocks() as demo:   # gr.Blocks：更灵活的布局方式
    # 6.1 添加标题和描述
    gr.Markdown("# CLIP Media Search (MVP)\n文本检索图片（可扩展到视频命中帧）+ Query 扩展融合")
    # "# " 表示一级标题
    # "\n" 换行，继续显示描述文字

    # 6.2 创建输入区域（第一行）
    with gr.Row():  # 横向排列的容器
        # 文本框：用于输入查询
        query = gr.Textbox(label="Query",   # 标签文字
                           placeholder="例如：南京夫子庙年味 / a dog on the beach" # 提示文本
                           )
    
    # 6.3 创建参数调整区域（第二行）
    with gr.Row():
        # 滑动条：控制返回结果数量
        topk = gr.Slider(5, 50,  # 最小值，最大值
                         value=20,  # 默认值
                         step=1,  # 步长
                         label="Top-K"  # 标签
                         ) 
        # 滑动条：控制扩展查询数量
        expand_n = gr.Slider(1, 10,  # 最小值，最大值
                              value=6,   # 默认值
                              step=1,   # 步长
                              label="Number of expanded prompts"   # 标签
                              )
        # 开关：是否用 LLM 改写/扩展查询（中英文都生成多条英文视觉描述）
        use_llm = gr.Checkbox(value=True, label="使用 LLM 改写查询")
        
    # 6.4 搜索按钮
    btn = gr.Button("Search")  # 按钮，显示文字"Search"

    # 6.5 结果展示区域
    # 画廊组件：用于显示图片网格
    gallery = gr.Gallery(
        label="Results", # 标签
          columns=4,  # 每行显示4列图片
          height=600  # 区域高度600像素
          )
    
    # 6.6 调试信息区域
    # 文本框：显示调试信息（只读）
    debug = gr.Textbox(label="Prompt debug (top1 score for each prompt)",  # 标签
                       lines=8  # 显示8行高度
                       )

    # 7. 绑定按钮点击事件
    # 当按钮被点击时，执行run函数

    btn.click(fn=run,  # 要执行的函数
              inputs=[query, topk, expand_n, use_llm],  # 输入参数：来自4个组件
              outputs=[gallery, debug]  # 输出结果：更新画廊和调试框
              )

# 8. 启动应用
if __name__ == "__main__":
    demo.launch()  # 启动Web服务器，默认地址：http://127.0.0.1:7860


