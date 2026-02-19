# 导入Gradio库：用于快速构建机器学习Web界面
import sys
from pathlib import Path

# 保证从项目根或 src 目录运行都能正确导入
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import gradio as gr
from src.search import search_with_fusion, search_by_image, search_video_clips

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


def run_image(image, topk: int):
    """以图搜图：上传一张图，返回库中最相似的一批图片。"""
    if image is None:
        return [], "请上传一张图片后再搜索。"
    results = search_by_image(image_input=image, topk=topk)
    gallery = []
    for r in results:
        path = r["path"]
        score = r["score"]
        if r.get("type") == "image":
            gallery.append((path, f"{score:.4f}  {path}"))
        else:
            gallery.append((path, f"{score:.4f}  {r.get('video_path','')} @ {r.get('timestamp_sec','')}"))
    status = f"以图搜图：共返回 {len(results)} 条结果。"
    return gallery, status


"""
基于文本检索视频关键帧并播放短视频片段。
返回：
- 对 Radio 的 update（更新 choices 列表）
- 默认播放的视频 clip_path
- 调试信息
"""
def run_video(query: str, topk: int, expand_n: int, use_llm: bool):
    if not query.strip():
        return gr.update(choices=[], value=None), None, ""
    results, prompt_scores, expand_source = search_video_clips(
        user_query=query,
        topk=topk,
        expand_n=expand_n,
        use_llm=use_llm,
    )

    labels = []
    clip_paths = []
    for idx, r in enumerate(results):
        vpath = r.get("video_path", "")
        ts = r.get("timestamp_sec", "")
        score = r.get("score", 0.0)
        label = f"{idx+1}. {vpath} @ {ts}s (score={score:.4f})"
        labels.append(label)
        # 优先使用截取的短 clip，失败则退回原视频；统一用绝对路径便于 Gradio 播放
        raw = r.get("clip_path") or vpath
        if raw:
            p = Path(raw)
            clip_paths.append(str((_root / raw).resolve() if not p.is_absolute() else p.resolve()))
        else:
            clip_paths.append(None)

    debug = f"[扩展方式] {expand_source}\n\n" + "\n".join([f"{s:.4f}\t{p}" for p, s in prompt_scores])
    if not labels:
        debug += "\n\n未命中任何视频帧。"
        return gr.update(choices=[], value=None), None, debug

    first_clip = clip_paths[0] if clip_paths else None
    return gr.update(choices=labels, value=None), first_clip, debug


# 6. 创建Gradio界面（三个 Tab：文本检索 + 以图搜图 + 捕捉视频关键帧）
with gr.Blocks() as demo:
    gr.Markdown("# CLIP Media Search (MVP)\n文本检索 / 以图搜图 / 捕捉视频关键帧 + Query 扩展融合")

    with gr.Tabs():
        # Tab1：文本检索
        with gr.Tab("文本检索"):
            with gr.Row():
                query = gr.Textbox(label="Query", placeholder="例如：南京夫子庙年味 / a dog on the beach")
            with gr.Row():
                topk = gr.Slider(5, 50, value=20, step=1, label="Top-K")
                expand_n = gr.Slider(1, 10, value=6, step=1, label="Number of expanded prompts")
                use_llm = gr.Checkbox(value=True, label="使用 LLM 改写查询")
            btn = gr.Button("Search")
            gallery = gr.Gallery(label="Results", columns=4, height=600)
            debug = gr.Textbox(label="Prompt debug (top1 score for each prompt)", lines=8)
            btn.click(fn=run, inputs=[query, topk, expand_n, use_llm], outputs=[gallery, debug])

        # Tab2：以图搜图
        with gr.Tab("以图搜图"):
            with gr.Row():
                image_in = gr.Image(label="上传图片", type="numpy")
            with gr.Row():
                topk_img = gr.Slider(5, 50, value=20, step=1, label="Top-K")
            btn_img = gr.Button("以图搜图")
            gallery_img = gr.Gallery(label="相似图片", columns=4, height=600)
            status_img = gr.Textbox(label="状态", lines=2)
            btn_img.click(fn=run_image, inputs=[image_in, topk_img], outputs=[gallery_img, status_img])

        # Tab3：捕捉视频关键帧
        with gr.Tab("捕捉视频关键帧"):
            with gr.Row():
                query_v = gr.Textbox(label="Query (文本搜视频)", placeholder="例如：打篮球的人 / 日落海滩")
            with gr.Row():
                topk_v = gr.Slider(1, 20, value=5, step=1, label="Top-K 视频帧")
                expand_n_v = gr.Slider(1, 10, value=6, step=1, label="Number of expanded prompts")
                use_llm_v = gr.Checkbox(value=True, label="使用 LLM 改写查询")

            btn_v = gr.Button("搜索视频关键帧")
            # 左侧：命中结果列表；右侧：视频播放器（左右高度尽量一致）
            with gr.Row(equal_height=True):
                choices_v = gr.Radio(label="命中视频帧列表", choices=[], interactive=True, scale=1)
                video_player = gr.Video(label="命中视频片段", height=360, scale=1)
            debug_v = gr.Textbox(label="Prompt debug (video)", lines=8)

            btn_v.click(
                fn=run_video,
                inputs=[query_v, topk_v, expand_n_v, use_llm_v],
                outputs=[choices_v, video_player, debug_v],
            )

            # 选择列表变化时，更新播放器
            def _update_video(selected_label, query, topk, expand_n, use_llm):
                # 简化实现：重新跑一遍检索，根据 label 找到对应 clip_path
                if not selected_label:
                    return gr.update()
                results, _, _ = search_video_clips(
                    user_query=query,
                    topk=topk,
                    expand_n=expand_n,
                    use_llm=use_llm,
                )
                for idx, r in enumerate(results):
                    vpath = r.get("video_path", "")
                    ts = r.get("timestamp_sec", "")
                    score = r.get("score", 0.0)
                    label = f"{idx+1}. {vpath} @ {ts}s (score={score:.4f})"
                    if label == selected_label:
                        raw = r.get("clip_path") or vpath
                        if not raw:
                            return None
                        p = Path(raw)
                        return str((_root / raw).resolve() if not p.is_absolute() else str(p.resolve()))
                return gr.update()

            choices_v.change(
                fn=_update_video,
                inputs=[choices_v, query_v, topk_v, expand_n_v, use_llm_v],
                outputs=[video_player],
            )

# 8. 启动应用
if __name__ == "__main__":
    # 允许 Gradio 访问项目下的 storage/clips、data/videos 等，否则返回的本地视频路径无法播放
    demo.launch(allowed_paths=[str(_root)])


