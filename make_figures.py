# -*- coding: utf-8 -*-
"""生成报告用图：曲线图、柱状图、视觉对比图。

输出到 results/ 目录。
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import dataset
from median_baseline import median_fixed
from median_adaptive import adaptive_median
from metrics import psnr, ssim

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

ORDER = ["固定3x3", "固定5x5", "固定7x7", "AMF(S=5)", "AMF(S=7)", "AMF(S=9)"]
STYLE = {
    "固定3x3": dict(marker="o", ls="--", color="#888888"),
    "固定5x5": dict(marker="s", ls="--", color="#c08030"),
    "固定7x7": dict(marker="^", ls="--", color="#a03030"),
    "AMF(S=5)": dict(marker="D", ls="-", color="#2a7ab0"),
    "AMF(S=7)": dict(marker="v", ls="-", color="#1f5fa0"),
    "AMF(S=9)": dict(marker="*", ls="-", color="#0d3b66"),
}


def load_summary():
    df = pd.read_csv(os.path.join(OUT, "summary.csv"))
    return df


def fig_curves():
    df = load_summary()
    for metric, ylabel, fname, tag in [
        ("psnr", "PSNR (dB)", "fig1_psnr_vs_density.png", "a"),
        ("ssim", "SSIM", "fig2_ssim_vs_density.png", "b"),
    ]:
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for m in ORDER:
            sub = df[df.method == m].sort_values("density")
            ax.plot(sub.density * 100, sub[metric], label=m, lw=1.8, ms=6, **STYLE[m])
        ax.set_xlabel("椒盐噪声密度（%）")
        ax.set_ylabel(ylabel)
        ax.set_title(f"({tag}) 不同方法在测试图集上的平均 {ylabel}")
        ax.grid(alpha=0.3, ls=":")
        ax.legend(fontsize=8, ncol=2, loc="lower left")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, fname))
        plt.close(fig)
        print("saved", fname)


def fig_runtime():
    rt = pd.read_csv(os.path.join(OUT, "runtime.csv"))
    rt = rt.set_index("method").loc[ORDER].reset_index()
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    colors = ["#888888", "#c08030", "#a03030", "#2a7ab0", "#1f5fa0", "#0d3b66"]
    bars = ax.bar(rt.method, rt.time_s * 1000, color=colors, width=0.62)
    for b, v in zip(bars, rt.time_s * 1000):
        ax.text(b.get_x() + b.get_width() / 2, v + 6, f"{v:.0f}", ha="center", fontsize=8)
    ax.set_ylabel("平均耗时（ms）")
    ax.set_title("(c) 各方法在 512×512 图像上的平均耗时（噪声密度 20%）")
    ax.grid(axis="y", alpha=0.3, ls=":")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_runtime.png"))
    plt.close(fig)
    print("saved fig3_runtime.png")


def metric_text(im, clean, name):
    """原始图自身比较会得到 PSNR=inf，单独标注。"""
    if name == "原始图像":
        return "（参考图像）"
    return f"PSNR={psnr(im, clean):.2f}dB SSIM={ssim(im, clean):.4f}"


def fig_visual():
    """视觉对比：20% 噪声下各方法效果 + 局部放大。"""
    imgs = dataset.load_images()
    clean = imgs["camera"]
    noisy = dataset.add_salt_pepper(clean, 0.20, seed=42)

    outs = {
        "原始图像": clean,
        "含20%椒盐噪声": noisy,
        "固定3x3": median_fixed(noisy, 3),
        "固定7x7": median_fixed(noisy, 7),
        "AMF(S=7)": adaptive_median(noisy, 7),
    }
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.5))
    for ax, (name, im) in zip(axes, outs.items()):
        ax.imshow(im, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{name}\n{metric_text(im, clean, name)}", fontsize=8)
        ax.axis("off")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.suptitle("(d) 视觉对比：512×512 camera，噪声密度 20%", fontsize=10, y=0.99)
    fig.savefig(os.path.join(OUT, "fig4_visual.png"))
    plt.close(fig)

    # 局部放大（取人脸眼睛区域）
    y0, y1, x0, x1 = 200, 280, 200, 280
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.5))
    for ax, (name, im) in zip(axes, outs.items()):
        ax.imshow(im[y0:y1, x0:x1], cmap="gray", vmin=0, vmax=255,
                  interpolation="nearest")
        ax.set_title(name, fontsize=8)
        ax.axis("off")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.suptitle("(e) 局部放大（80×80 区域）：观察噪声残留与细节丢失", fontsize=10, y=0.99)
    fig.savefig(os.path.join(OUT, "fig5_zoom.png"))
    plt.close(fig)
    print("saved fig4_visual.png, fig5_zoom.png")


def fig_detail():
    """无噪声情形：检验方法是否破坏真实细节（AMF 的核心优势）。"""
    imgs = dataset.load_images()
    clean = imgs["camera"]
    outs = {
        "原始图像": clean,
        "固定3x3": median_fixed(clean, 3),
        "固定7x7": median_fixed(clean, 7),
        "AMF(S=7)": adaptive_median(clean, 7),
    }
    y0, y1, x0, x1 = 150, 250, 150, 250
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for j, (name, im) in enumerate(outs.items()):
        axes[0, j].imshow(im, cmap="gray", vmin=0, vmax=255)
        sub = "（参考图像）" if name == "原始图像" else f"PSNR={psnr(im, clean):.2f}dB"
        axes[0, j].set_title(f"{name}\n{sub}", fontsize=9)
        axes[0, j].axis("off")
        axes[1, j].imshow(im[y0:y1, x0:x1], cmap="gray", vmin=0, vmax=255,
                          interpolation="nearest")
        axes[1, j].set_title("局部放大", fontsize=8)
        axes[1, j].axis("off")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.suptitle("(f) 无噪声图像：固定窗口会模糊真实细节，AMF 基本保真", fontsize=11, y=0.99)
    fig.savefig(os.path.join(OUT, "fig6_detail_preserve.png"))
    plt.close(fig)
    print("saved fig6_detail_preserve.png")


if __name__ == "__main__":
    fig_curves()
    fig_runtime()
    fig_visual()
    fig_detail()
    print("ALL FIGURES DONE ->", OUT)
