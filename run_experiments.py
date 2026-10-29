# -*- coding: utf-8 -*-
"""实验主程序：噪声密度 × 图像 × 方法 的全矩阵评测。

输出：
    results/results.csv        逐条结果（长表）
    results/summary.csv        按方法汇总（对 4 张图取平均）
    results/runtime.csv        各方法平均耗时
"""
import os
import sys
import numpy as np
import pandas as pd

import dataset
from median_baseline import median_fixed
from median_adaptive import adaptive_median
from linear_filters import mean_filter, gaussian_filter
from metrics import psnr, ssim, timeit

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)

DENSITIES = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]

# 方法名 -> (可调用对象, 说明)
METHODS = {
    # ---- 线性平滑（对照）----
    "均值3x3": (lambda x: mean_filter(x, 3), "线性平滑：3x3 邻域平均"),
    "高斯(σ=1.5)": (lambda x: gaussian_filter(x, 1.5), "线性平滑：高斯 sigma=1.5"),
    # ---- 固定窗口中值（对照）----
    "固定3x3": (lambda x: median_fixed(x, 3), "固定窗口 k=3"),
    "固定5x5": (lambda x: median_fixed(x, 5), "固定窗口 k=5"),
    "固定7x7": (lambda x: median_fixed(x, 7), "固定窗口 k=7"),
    # ---- 本文方法 ----
    "AMF(S=5)": (lambda x: adaptive_median(x, 5), "自适应，最大窗口 5"),
    "AMF(S=7)": (lambda x: adaptive_median(x, 7), "自适应，最大窗口 7"),
    "AMF(S=9)": (lambda x: adaptive_median(x, 9), "自适应，最大窗口 9"),
}


def main():
    imgs = dataset.load_images()
    rows = []
    for name, clean in imgs.items():
        for d in DENSITIES:
            noisy = clean if d == 0.0 else dataset.add_salt_pepper(clean, d, seed=42)
            for mname, (fn, _desc) in METHODS.items():
                t, out = timeit(fn, noisy, repeat=3, warmup=1)
                rows.append({
                    "image": name,
                    "density": d,
                    "method": mname,
                    "psnr": psnr(out, clean),
                    "ssim": ssim(out, clean),
                    "time_s": t,
                })
                print(f"  {name:6s} d={d:.2f} {mname:9s} "
                      f"PSNR={rows[-1]['psnr']:6.2f} SSIM={rows[-1]['ssim']:.4f} "
                      f"t={t*1000:7.1f} ms", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "results.csv"), index=False, encoding="utf-8-sig")

    # 按方法汇总（对 4 张图取平均）
    summ = (df.groupby(["method", "density"], as_index=False)
              .agg(psnr=("psnr", "mean"), ssim=("ssim", "mean"),
                   time_s=("time_s", "mean")))
    summ.to_csv(os.path.join(OUT, "summary.csv"), index=False, encoding="utf-8-sig")

    # 耗时（只统计 d=0.20 这一档，避免混入不同噪声的差异）
    rt = (df[df.density == 0.20].groupby("method", as_index=False)
            .agg(time_s=("time_s", "mean")))
    rt = rt.sort_values("time_s")
    rt.to_csv(os.path.join(OUT, "runtime.csv"), index=False, encoding="utf-8-sig")

    # 终端汇总表
    print("\n" + "=" * 78)
    print("平均 PSNR (dB) / SSIM —— 4 张测试图平均")
    print("=" * 78)
    piv_p = summ.pivot(index="method", columns="density", values="psnr")
    piv_s = summ.pivot(index="method", columns="density", values="ssim")
    order = list(METHODS.keys())
    print("PSNR:")
    print(piv_p.loc[order].round(2).to_string())
    print("\nSSIM:")
    print(piv_s.loc[order].round(4).to_string())
    print("\n平均耗时 (d=0.20):")
    for _, r in rt.iterrows():
        print(f"  {r.method:9s} {r.time_s*1000:8.1f} ms")
    print(f"\n结果已写入 {OUT}")


if __name__ == "__main__":
    main()
