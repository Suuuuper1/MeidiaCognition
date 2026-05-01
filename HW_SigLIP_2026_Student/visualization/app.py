"""
SigLIP 图文相似度演示
====================
启动方式：
    cd homework_new/
    python app.py

访问：http://localhost:7860
"""

import os
import sys
import random
import logging
import base64
import numpy as np
from io import BytesIO

import torch
import gradio as gr
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("app")

from config import HEATMAP_MAX_N, N_SAMPLES
from model.siglip_wrapper import SigLIPWrapper
from data.flickr8k_loader import load_flickr8k
from viz.v1_similarity_matrix import create_similarity_heatmap, pil_to_b64

# ═══════════════════════════════════════════════════════════════════════════════
#  全局初始化
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("  SigLIP 图文相似度演示 — 正在初始化...")
print("=" * 60)

print("[1/2] 加载 SigLIP 模型 (google/siglip-base-patch16-224)...")
MODEL = SigLIPWrapper()

print(f"[2/2] 加载 Flickr8K 训练集（最多 {N_SAMPLES} 样本）...")
FLICKR_SAMPLES = load_flickr8k(split="train", n_samples=N_SAMPLES)

print("=" * 60)
print(f"  初始化完成！共 {len(FLICKR_SAMPLES)} 张图像")
print("  访问 http://localhost:7860")
print("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════════
#  回调函数
# ═══════════════════════════════════════════════════════════════════════════════

def _make_gallery_html(images: list, captions: list) -> str:
    """生成当前批次的图文展示卡片。"""
    items = []
    for i, (img, cap) in enumerate(zip(images, captions)):
        b64 = pil_to_b64(img, size=(160, 120))
        items.append(f"""
        <div style="width:190px;flex-shrink:0;background:#1e2536;
                    border:1px solid #2d3d5e;border-radius:8px;overflow:hidden">
            <div style="position:relative">
                <img src="data:image/jpeg;base64,{b64}"
                     style="width:100%;height:120px;object-fit:cover;display:block">
                <div style="position:absolute;top:5px;left:5px;
                            background:rgba(0,0,0,.7);color:#79b8ff;
                            font-size:12px;font-weight:700;
                            padding:2px 8px;border-radius:10px">#{i}</div>
            </div>
            <div style="padding:8px 10px;font-size:11px;color:#b0bec5;
                        line-height:1.5;min-height:52px">{cap}</div>
        </div>""")

    return f"""
    <div style="margin-top:20px">
        <div style="font-size:13px;color:#8b949e;margin-bottom:10px;font-weight:600;
                    letter-spacing:.3px">当前采样图文对（共 {len(images)} 个）</div>
        <div style="display:flex;flex-wrap:wrap;gap:10px">{''.join(items)}</div>
    </div>"""


def cb_resample(n_pairs: int):
    """随机采样新一批图文对，生成热力图 + 统计 + 图文卡片。"""
    n     = int(n_pairs)
    batch = random.sample(FLICKR_SAMPLES, min(n, len(FLICKR_SAMPLES)))
    images   = [s["image"]   for s in batch]
    captions = [s["caption"] for s in batch]

    # 热力图
    fig = create_similarity_heatmap(images, captions, MODEL)

    # 统计
    img_emb = MODEL.encode_images(images)
    txt_emb = MODEL.encode_texts(captions)
    sim     = MODEL.compute_similarity(img_emb, txt_emb).cpu().float().numpy()

    diag     = np.diag(sim)
    mask_off = ~np.eye(n, dtype=bool)
    off_diag = sim[mask_off]
    ratio    = diag.mean() / max(off_diag.mean(), 1e-8)
    ls, lb   = MODEL.get_logit_params()

    stat_html = f"""
    <div style="display:flex;gap:12px;flex-wrap:wrap;margin-top:16px;font-family:sans-serif">
        <div style="background:#161b22;border-radius:10px;padding:14px 20px;
                    border:1px solid #30363d;text-align:center;min-width:130px">
            <div style="font-size:1.6em;font-weight:700;color:#2ecc71">{diag.mean():.4f}</div>
            <div style="font-size:12px;color:#8b949e;margin-top:4px">正样本对均值<br>（对角线）</div>
        </div>
        <div style="background:#161b22;border-radius:10px;padding:14px 20px;
                    border:1px solid #30363d;text-align:center;min-width:130px">
            <div style="font-size:1.6em;font-weight:700;color:#e74c3c">{off_diag.mean():.4f}</div>
            <div style="font-size:12px;color:#8b949e;margin-top:4px">负样本对均值<br>（非对角线）</div>
        </div>
        <div style="background:#161b22;border-radius:10px;padding:14px 20px;
                    border:1px solid #30363d;text-align:center;min-width:130px">
            <div style="font-size:1.6em;font-weight:700;color:#f39c12">{ratio:.2f}×</div>
            <div style="font-size:12px;color:#8b949e;margin-top:4px">正/负<br>对比倍数</div>
        </div>
        <div style="background:#161b22;border-radius:10px;padding:14px 20px;
                    border:1px solid #30363d;text-align:center;min-width:130px">
            <div style="font-size:1.6em;font-weight:700;color:#58a6ff">{np.exp(ls):.1f}</div>
            <div style="font-size:12px;color:#8b949e;margin-top:4px">logit_scale<br>exp(ln s)</div>
        </div>
        <div style="background:#161b22;border-radius:10px;padding:14px 20px;
                    border:1px solid #30363d;text-align:center;min-width:130px">
            <div style="font-size:1.6em;font-weight:700;color:#9b59b6">{lb:.1f}</div>
            <div style="font-size:12px;color:#8b949e;margin-top:4px">logit_bias</div>
        </div>
    </div>"""

    gallery_html = _make_gallery_html(images, captions)
    return fig, stat_html, gallery_html


# ═══════════════════════════════════════════════════════════════════════════════
#  样式
# ═══════════════════════════════════════════════════════════════════════════════

_CSS = """
.gradio-container { background: #1a1f2e !important; min-height: 100vh; }
.gradio-container, .gradio-container p, .gradio-container span,
.gradio-container div, .gradio-container label,
.gradio-container h1, .gradio-container h2, .gradio-container h3 {
    color: #e8ecf0 !important;
}
#banner {
    background: linear-gradient(90deg, #1a2744 0%, #1e3a5f 50%, #1a2744 100%);
    border: 1px solid #2d4a7a; border-radius: 12px;
    padding: 28px 36px; text-align: center; margin-bottom: 20px;
    box-shadow: 0 4px 24px rgba(0,0,0,.4);
}
#banner h1 {
    font-size: 2em !important; color: #79b8ff !important;
    margin: 0 0 8px !important; text-shadow: 0 0 20px rgba(121,184,255,.4);
}
#banner p  { color: #b0bec5 !important; margin: 0 !important; font-size: 14px !important; }
#banner .badge {
    display: inline-block; background: #263650; border: 1px solid #3a5278;
    border-radius: 20px; padding: 3px 12px;
    font-size: 12px; color: #90caf9 !important; margin: 6px 3px 0;
}
.prose *, .markdown * { color: #dce3ec !important; line-height: 1.7 !important; }
.prose h1, .prose h2, .prose h3,
.markdown h1, .markdown h2, .markdown h3 {
    color: #79b8ff !important; border-bottom: 1px solid #2d3d5e; padding-bottom: 4px;
}
.prose code, .markdown code {
    background: #263650 !important; color: #f0a070 !important;
    border-radius: 4px; padding: 1px 6px; font-size: 0.88em;
}
button.primary, button[variant="primary"] {
    background: linear-gradient(135deg, #1e5f9a, #2980b9) !important;
    border: none !important; color: #ffffff !important; font-weight: 600 !important;
    border-radius: 8px !important; box-shadow: 0 3px 12px rgba(41,128,185,.4) !important;
}
button.primary:hover, button[variant="primary"]:hover {
    background: linear-gradient(135deg, #2980b9, #3498db) !important;
    transform: translateY(-1px) !important;
}
input[type="range"] { accent-color: #4a90d9 !important; }
.gr-panel, .gr-box, .gr-form, .gr-block {
    background: #1e2536 !important; border: 1px solid #2d3d5e !important;
    border-radius: 10px !important;
}
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #1a1f2e; }
::-webkit-scrollbar-thumb { background: #3a5278; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #4a90d9; }
"""

_BANNER = """
<div id="banner">
    <h1>SigLIP 图文相似度演示</h1>
    <p>基于 <b>google/siglip-base-patch16-224</b> 预训练模型 &nbsp;|&nbsp; Flickr8K 训练集</p>
    <div style="margin-top:10px">
        <span class="badge">Sigmoid 相似度（非 Softmax）</span>
        <span class="badge">ViT-B/16 图像编码器</span>
        <span class="badge">Text Transformer</span>
        <span class="badge">Flickr8K 训练集</span>
    </div>
</div>
"""

# ═══════════════════════════════════════════════════════════════════════════════
#  构建界面
# ═══════════════════════════════════════════════════════════════════════════════

with gr.Blocks(
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.blue,
        neutral_hue=gr.themes.colors.slate,
        font=gr.themes.GoogleFont("Inter"),
        text_size=gr.themes.sizes.text_md,
    ).set(
        body_background_fill="#1a1f2e",
        body_text_color="#e8ecf0",
        block_background_fill="#1e2536",
        block_label_text_color="#90caf9",
        block_title_text_color="#79b8ff",
        input_background_fill="#263650",
        input_border_color="#3a5278",
        button_primary_background_fill="linear-gradient(135deg,#1e5f9a,#2980b9)",
        button_primary_text_color="#ffffff",
        slider_color="#4a90d9",
    ),
    css=_CSS,
    title="SigLIP 图文相似度演示",
) as demo:

    gr.HTML(_BANNER)

    gr.Markdown("""
    ## 图文交叉相似度矩阵

    从 Flickr8K 训练集随机采样若干图文对，计算并可视化 **N×N** 的 Sigmoid 相似度矩阵。

    - **列**：采样的图像（上方附缩略图，编号 #0, #1, ...）
    - **行**：对应的文本描述（截断后显示在左侧）
    - **对角线（绿框）**：真实匹配对，相似度应明显高于非对角线
    - **非对角线**：不匹配的图文组合，相似度应接近 0
    - **鼠标悬停**：查看该格子对应的图像缩略图和完整描述

    每次点击「随机采样」会从 Flickr8K 训练集中抽取全新一批图文对。
    """)

    with gr.Row():
        n_slider    = gr.Slider(4, HEATMAP_MAX_N, value=8, step=2,
                                label="图文对数量 N（矩阵大小 N×N）")
        sample_btn  = gr.Button("随机采样并生成热力图", variant="primary", scale=1)

    heatmap_out = gr.Plot()
    metrics_out = gr.HTML()
    gallery_out = gr.HTML()

    sample_btn.click(cb_resample, inputs=[n_slider],
                     outputs=[heatmap_out, metrics_out, gallery_out])

    gr.HTML("""
    <div style="background:#1c2333;border-radius:8px;padding:14px 18px;
                border-left:4px solid #4a90d9;font-size:13px;color:#c9d1d9;margin-top:8px">
        <b>关键区别</b>：CLIP 的对比损失使用 <code>softmax</code>——每张图在 batch 内
        与所有文本竞争（多分类）。SigLIP 使用 <code>sigmoid</code>——每个 (图,文) 对
        独立判断是否匹配（二分类），对 batch 大小更鲁棒，支持每张图有多个正例。
    </div>
    """)

# ═══════════════════════════════════════════════════════════════════════════════
#  启动
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    demo.queue(max_size=4).launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=True,
    )
