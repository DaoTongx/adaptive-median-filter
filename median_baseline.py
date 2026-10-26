# -*- coding: utf-8 -*-
"""【基线实现】固定窗口中值滤波。

这是大作业的"改动前"版本：窗口大小 k 在调用时固定，对整幅图所有像素一视同仁。
本文件在报告里对应"代码修改描述"的 before 状态。
"""
import numpy as np
from scipy import ndimage


def median_fixed(img, k=3):
    """固定窗口中值滤波（工程实现）。

    参数
    ----
    img : uint8 二维数组
    k   : 窗口边长（奇数），3/5/7 ...
    """
    if k % 2 == 0 or k < 3:
        raise ValueError("k 必须是 >=3 的奇数")
    return ndimage.median_filter(img, size=k, mode="reflect")


def median_fixed_naive(img, k=3):
    """固定窗口中值滤波（教学版，逐窗口取中值）。

    用滑动窗口视图一次构造所有窗口，再沿窗口两个轴取中值。
    代码直观、便于对照，但内存占用随 k^2 增长，因此工程版用 median_fixed。
    """
    if k % 2 == 0 or k < 3:
        raise ValueError("k 必须是 >=3 的奇数")
    r = (k - 1) // 2
    # 注意：scipy 的 mode="reflect" 是"半样本对称"，等价于 numpy 的 "symmetric"
    pad = np.pad(img.astype(np.float64), r, mode="symmetric")
    win = np.lib.stride_tricks.sliding_window_view(pad, (k, k))
    med = np.median(win, axis=(-2, -1))
    return np.clip(med, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    import dataset
    from metrics import psnr, ssim

    imgs = dataset.load_images()
    clean = imgs["camera"]
    noisy = dataset.add_salt_pepper(clean, 0.2)
    print("固定窗口中值滤波（噪声密度 20%）")
    for k in (3, 5, 7):
        out = median_fixed(noisy, k)
        print(f"  k={k}: PSNR={psnr(out, clean):6.2f} dB  SSIM={ssim(out, clean):.4f}")
    # 两种实现一致性校验
    a = median_fixed(noisy, 5)
    b = median_fixed_naive(noisy, 5)
    print(f"  工程版与教学版输出是否一致: {np.array_equal(a, b)}")
