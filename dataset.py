# -*- coding: utf-8 -*-
"""数据集：测试图像 + 合成椒盐噪声。

所有图像均来自 scikit-image 自带数据，无需联网下载。
"""
import numpy as np
from skimage import data, transform

IMG_SIZE = 512          # 统一到 512x512，便于横向比较


def load_images():
    """返回 {名称: uint8 灰度图 512x512}。"""
    raw = {
        "camera": data.camera(),     # 人脸 + 建筑，含大量细节与平坦区
        "coins": data.coins(),       # 金属硬币，边缘锐利
        "moon": data.moon(),         # 月球表面，弱纹理平坦区多
        "text": data.text(),         # 文字页，细笔画（强细节结构）
    }
    out = {}
    for name, im in raw.items():
        if im.ndim == 3:
            im = im.mean(axis=2)
        im = im.astype(np.float64)
        if im.shape != (IMG_SIZE, IMG_SIZE):
            im = transform.resize(im, (IMG_SIZE, IMG_SIZE),
                                  order=1, preserve_range=True, anti_aliasing=True)
        out[name] = np.clip(im, 0, 255).astype(np.uint8)
    return out


def add_salt_pepper(img, density, seed=0):
    """按密度 density 加入椒盐噪声（黑白各占一半）。

    density 是"被污染像素的比例"，例如 0.1 表示 10% 的像素被置为 0 或 255。
    """
    rng = np.random.default_rng(seed)
    noisy = img.copy()
    u = rng.random(img.shape)
    noisy[u < density / 2.0] = 0        # 椒（黑）
    noisy[u > 1.0 - density / 2.0] = 255  # 盐（白）
    return noisy


def add_gaussian(img, sigma, seed=0):
    """高斯噪声，作为"非脉冲噪声"的对照（自适应中值滤波对它无效）。"""
    rng = np.random.default_rng(seed)
    noisy = img.astype(np.float64) + rng.normal(0, sigma, img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    imgs = load_images()
    for k, v in imgs.items():
        print(f"{k:8s} {v.shape} dtype={v.dtype} 均值={v.mean():.1f}")
    n = add_salt_pepper(imgs["camera"], 0.2)
    changed = int((n != imgs["camera"]).sum())
    print(f"camera 加 20% 椒盐噪声：实际改变 {changed} 像素 "
          f"({changed / imgs['camera'].size * 100:.2f}%)")
