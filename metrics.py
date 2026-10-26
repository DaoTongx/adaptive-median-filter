# -*- coding: utf-8 -*-
"""评价指标与计时工具。"""
import time
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def psnr(a, b):
    """峰值信噪比（dB）。a 为待评价图，b 为参考图，data_range=255。"""
    return float(peak_signal_noise_ratio(b, a, data_range=255))


def ssim(a, b):
    """结构相似度。"""
    return float(structural_similarity(b, a, data_range=255))


def timeit(fn, *args, repeat=3, warmup=1, **kwargs):
    """返回 (中位数耗时秒, 输出)。

    先空跑 warmup 次，再计时 repeat 次取中位数，避免首次调用的初始化开销
    掩盖真实性能。
    """
    out = None
    for _ in range(warmup):
        out = fn(*args, **kwargs)
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), out
