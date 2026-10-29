# -*- coding: utf-8 -*-
"""对照组：线性平滑滤波。

均值滤波与高斯滤波是图像平滑的经典方法，用来对照说明：
1) 线性平滑对脉冲噪声的效果远不如中值类方法；
2) 线性平滑会整体模糊边缘，而中值类方法能保留边缘。

本文件与 median_baseline.py 一起构成"对照方法"部分。
"""
import numpy as np
from scipy import ndimage


def mean_filter(img, k=3):
    """k×k 均值滤波（邻域平均）。

    每个像素取邻域内所有像素的算术平均值。对高斯噪声有效，
    但对椒盐噪声只是把脉冲的能量摊到邻域里，去不干净。
    """
    if k % 2 == 0 or k < 3:
        raise ValueError("k 必须是 >=3 的奇数")
    src = img.astype(np.float64)
    out = ndimage.uniform_filter(src, size=k, mode="reflect")
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def gaussian_filter(img, sigma=1.5):
    """高斯滤波。

    按二维高斯核加权平均，权重随距离衰减，比等权均值略好，
    但同样是线性平滑，会把脉冲噪声抹成灰斑、把边缘糊开。
    """
    src = img.astype(np.float64)
    out = ndimage.gaussian_filter(src, sigma=sigma, mode="reflect")
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


if __name__ == "__main__":
    import dataset
    from metrics import psnr, ssim

    imgs = dataset.load_images()
    clean = imgs["camera"]
    noisy = dataset.add_salt_pepper(clean, 0.20, seed=42)
    print("线性平滑在 20% 椒盐噪声上的表现：")
    for name, fn in [("均值 3x3", lambda x: mean_filter(x, 3)),
                     ("高斯 σ=1.5", lambda x: gaussian_filter(x, 1.5))]:
        out = fn(noisy)
        print(f"  {name:12s} PSNR={psnr(out, clean):6.2f} dB  SSIM={ssim(out, clean):.4f}")
    print("\n对照：固定 3x3 中值")
    from median_baseline import median_fixed
    out = median_fixed(noisy, 3)
    print(f"  {'固定 3x3':12s} PSNR={psnr(out, clean):6.2f} dB  SSIM={ssim(out, clean):.4f}")
