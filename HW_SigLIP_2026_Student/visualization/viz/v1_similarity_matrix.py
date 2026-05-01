"""
图文交叉相似度矩阵热力图
========================
展示 SigLIP 图文对齐的核心结果：
  - 对角线 = 正样本对（图像与其对应描述，应为高值）
  - 非对角线 = 负样本对（图像与不匹配描述，应接近 0）
  - X 轴缩略图 + Y 轴文本描述标签
  - 鼠标悬停：查看对应图像 + 完整描述
"""

import base64
import numpy as np
import torch
import plotly.graph_objects as go
from PIL import Image
from io import BytesIO
from typing import List


def pil_to_b64(img: Image.Image, size: tuple = (80, 80), quality: int = 82) -> str:
    """PIL 图像 → base64 JPEG 字符串（供外部调用）。"""
    img = img.convert("RGB").resize(size, Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def _truncate(text: str, max_len: int = 28) -> str:
    return text[:max_len] + "…" if len(text) > max_len else text


def create_similarity_heatmap(
    images: List[Image.Image],
    captions: List[str],
    model,
) -> go.Figure:
    """
    生成交互式图文相似度矩阵热力图。

    布局：
      - 列（X 轴）= 图像，轴上方附缩略图
      - 行（Y 轴）= 对应文本描述（截断显示），不重复贴图
      - 颜色 = sigmoid 相似度（0~1）
      - 绿框 = 正样本对（对角线）
      - 悬停 = 图像缩略图 + 完整描述

    Returns:
        go.Figure（可直接传给 gr.Plot）
    """
    n = len(images)
    assert len(captions) == n

    # ── Step 1: 计算相似度矩阵 ────────────────────────────────────────────────
    img_embeds = model.encode_images(images)
    txt_embeds = model.encode_texts(captions)
    sim_matrix = model.compute_similarity(img_embeds, txt_embeds)
    sim_np     = sim_matrix.cpu().float().numpy()   # [N_img, N_text]

    # 热力图 z[row=text, col=image]
    sim_display = sim_np.T

    # ── Step 2: 图片缩略图 ─────────────────────────────────────────────────────
    thumb_b64 = [pil_to_b64(img) for img in images]

    # ── Step 3: customdata（hover 用）─────────────────────────────────────────
    custom = np.empty((n, n), dtype=object)
    for row_i in range(n):
        for col_j in range(n):
            custom[row_i, col_j] = [
                thumb_b64[col_j],
                captions[col_j],          # 图像对应描述（完整）
                captions[row_i],          # 行对应描述（完整）
            ]

    hover_tmpl = (
        "<b>Sigmoid 相似度: %{z:.4f}</b><br>"
        "<img src='data:image/jpeg;base64,%{customdata[0]}' "
        "style='width:110px;height:110px;object-fit:cover;border-radius:6px'><br>"
        "<span style='color:#aaa;font-size:11px'>图像描述：%{customdata[1]}</span><br><br>"
        "<b>文本行：</b><br>"
        "<span style='color:#c9d1d9'>%{customdata[2]}</span>"
        "<extra></extra>"
    )

    # ── Step 4: 轴标签 ────────────────────────────────────────────────────────
    x_labels = [f"#{j}" for j in range(n)]
    # Y 轴用截断的描述文字，而非"文本 i"——更直观地展示对应关系
    y_labels = [_truncate(cap, 30) for cap in captions]

    # ── Step 5: Heatmap ───────────────────────────────────────────────────────
    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        z=sim_display,
        x=x_labels,
        y=y_labels,
        colorscale=[
            [0.00, "#0d1117"],
            [0.15, "#0a2a4a"],
            [0.35, "#1a5276"],
            [0.55, "#2980b9"],
            [0.75, "#f39c12"],
            [1.00, "#e74c3c"],
        ],
        zmin=0.0, zmax=1.0,
        customdata=custom,
        hovertemplate=hover_tmpl,
        showscale=True,
        colorbar=dict(
            title=dict(text="Sigmoid<br>相似度", font=dict(size=12, color="white")),
            tickformat=".2f",
            tickfont=dict(color="white"),
            thickness=18,
            len=0.75,
            x=1.02,
        ),
    ))

    # ── Step 6: 对角线高亮（正样本对）───────────────────────────────────────
    for i in range(n):
        fig.add_shape(
            type="rect",
            x0=i - 0.5, x1=i + 0.5,
            y0=i - 0.5, y1=i + 0.5,
            line=dict(color="#2ecc71", width=2.5),
            fillcolor="rgba(0,0,0,0)",
        )

    # ── Step 7: X 轴图像缩略图（列方向，上方）────────────────────────────────
    for j, b64 in enumerate(thumb_b64):
        fig.add_layout_image(dict(
            source=f"data:image/jpeg;base64,{b64}",
            x=j, y=n + 0.9,
            xref="x", yref="y",
            sizex=0.82, sizey=0.82,
            xanchor="center", yanchor="middle",
            layer="above",
        ))

    # ── Step 8: 统计注释 ──────────────────────────────────────────────────────
    diag_vals    = sim_np.diagonal()
    mask_offdiag = ~np.eye(n, dtype=bool)
    diag_mean    = diag_vals.mean()
    offdiag_mean = sim_np[mask_offdiag].mean()
    logit_scale, logit_bias = model.get_logit_params()

    note_text = (
        f"<b>SigLIP：sigmoid 二分类（非 CLIP 的 softmax 多分类）</b><br>"
        f"相似度 = sigmoid( exp({logit_scale:.2f}) × cos_sim + ({logit_bias:.1f}) )<br>"
        f"绿框 = 正样本对（对角线）| "
        f"正样本均值 <b style='color:#2ecc71'>{diag_mean:.4f}</b> | "
        f"负样本均值 <b style='color:#e74c3c'>{offdiag_mean:.4f}</b>"
    )
    fig.add_annotation(
        text=note_text,
        xref="paper", yref="paper",
        x=0.5, y=-0.13,
        showarrow=False,
        font=dict(size=11, color="#c9d1d9"),
        align="center",
        bgcolor="#1c2333",
        bordercolor="#30363d",
        borderpad=8,
        borderwidth=1,
    )

    # ── Step 9: 布局 ──────────────────────────────────────────────────────────
    # Y 轴文字标签较长，左侧留更多空间
    left_margin = min(300, max(160, 8 * 30))

    fig.update_layout(
        title=dict(
            text="图文交叉相似度矩阵（SigLIP Sigmoid 相似度）",
            font=dict(size=18, color="white"),
            x=0.5, pad=dict(b=20),
        ),
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        font=dict(color="white", size=11),
        xaxis=dict(
            showticklabels=True,
            tickfont=dict(size=12, color="#79b8ff"),
            showgrid=False,
            range=[-0.6, n + 0.6],
        ),
        yaxis=dict(
            showticklabels=True,
            tickfont=dict(size=10, color="#b0bec5"),
            showgrid=False,
            range=[-0.6, n + 1.4],
            automargin=True,
        ),
        margin=dict(l=left_margin, r=80, t=160, b=110),
        height=max(560, 60 * n + 220),
        hoverlabel=dict(
            bgcolor="#1c2333",
            bordercolor="#4a90d9",
            font_size=12,
        ),
    )

    return fig
