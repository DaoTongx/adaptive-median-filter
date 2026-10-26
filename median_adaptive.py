# -*- coding: utf-8 -*-
"""【改进实现】自适应中值滤波（Adaptive Median Filter, AMF）。

这是大作业的"改动后"版本，对应报告第 4 项"代码修改描述"的 after 状态。

算法来源：Gonzalez & Woods《数字图像处理》中的经典自适应中值滤波。
与固定窗口中值滤波的区别：
    固定窗口 —— 全图统一用 k×k 窗口，高噪声去不干净、低噪声糊掉细节；
    自适应   —— 每个像素独立地从 3×3 起逐步放大窗口，一旦判定"当前窗口可给出
                可靠中值"就停止放大，并且若中心像素本身不是脉冲就原样保留。

算法两级判据（z_min/z_max/z_med 为当前窗口的最小/最大/中值，z_xy 为中心像素）：

    第 A 级：z_min < z_med < z_max ?
        成立   -> 中值不是极值，说明窗口里有可用信息，转第 B 级
        不成立 -> 中值仍是脉冲（窗口太小），放大窗口；若已达 S_max 则输出 z_med
    第 B 级：z_min < z_xy < z_max ?
        成立   -> 中心像素不是脉冲，原样输出 z_xy（保细节）
        不成立 -> 中心像素是脉冲，输出 z_med（去噪）
"""
import numpy as np
from scipy import ndimage


def adaptive_median(img, s_max=7):
    """自适应中值滤波。

    参数
    ----
    img   : uint8 二维数组
    s_max : 允许的最大窗口边长（奇数）。3 表示退化为普通 3x3 中值滤波。

    返回
    ----
    uint8 二维数组
    """
    if s_max % 2 == 0 or s_max < 3:
        raise ValueError("s_max 必须是 >=3 的奇数")

    src = img.astype(np.float64)
    out = src.copy()
    # undecided[i,j] = True 表示该像素在第 A 级尚未通过，还需放大窗口
    undecided = np.ones(src.shape, dtype=bool)

    z_med_last = None
    r_max = (s_max - 1) // 2

    for r in range(1, r_max + 1):
        k = 2 * r + 1
        z_min = ndimage.minimum_filter(src, size=k, mode="reflect")
        z_max = ndimage.maximum_filter(src, size=k, mode="reflect")
        z_med = ndimage.median_filter(src, size=k, mode="reflect")
        z_med_last = z_med

        # 第 A 级：中值不是窗口极值 —— 说明该窗口的中值可信
        pass_A = (z_med > z_min) & (z_med < z_max)
        decide = undecided & pass_A          # 本层可以定案的像素

        # 第 B 级：区分"中心是脉冲"与"中心是真实细节"
        center_ok = (src > z_min) & (src < z_max)
        keep_src = decide & center_ok        # 保留原值
        use_med = decide & ~center_ok        # 用中值替换

        out[keep_src] = src[keep_src]
        out[use_med] = z_med[use_med]

        undecided &= ~pass_A                 # 通过的像素不再参与后续层

    # 到达 S_max 仍未定案：按算法规定输出最大窗口下的中值
    if z_med_last is not None:
        out[undecided] = z_med_last[undecided]

    return np.clip(out, 0, 255).astype(np.uint8)


def adaptive_median_naive(img, s_max=7):
    """逐像素版本（与向量化版等价，仅用于正确性对照，速度很慢）。"""
    from scipy import ndimage as ndi

    src = img.astype(np.float64)
    # scipy 的 mode="reflect" 等价于 numpy 的 "symmetric"
    padded = np.pad(src, (s_max - 1) // 2, mode="symmetric")
    out = src.copy()
    h, w = src.shape
    off = (s_max - 1) // 2

    for i in range(h):
        for j in range(w):
            z_xy = src[i, j]
            for r in range(1, (s_max - 1) // 2 + 1):
                win = padded[i + off - r:i + off + r + 1,
                             j + off - r:j + off + r + 1]
                z_min, z_max, z_med = win.min(), win.max(), np.median(win)
                if z_min < z_med < z_max:
                    out[i, j] = z_xy if z_min < z_xy < z_max else z_med
                    break
            else:
                win = padded[i + off - r:i + off + r + 1,
                             j + off - r:j + off + r + 1]
                out[i, j] = np.median(win)
    return np.clip(out, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    import dataset
    from median_baseline import median_fixed
    from metrics import psnr, ssim

    imgs = dataset.load_images()
    clean = imgs["camera"]

    print("=== 无噪声图像上的保细节能力（越接近原图越好）===")
    for name, fn in [("固定3x3", lambda x: median_fixed(x, 3)),
                     ("固定7x7", lambda x: median_fixed(x, 7)),
                     ("AMF(S_max=7)", lambda x: adaptive_median(x, 7))]:
        out = fn(clean)
        print(f"  {name:14s} PSNR={psnr(out, clean):6.2f} dB  SSIM={ssim(out, clean):.4f}")

    print("\n=== 20% 椒盐噪声上的去噪能力 ===")
    noisy = dataset.add_salt_pepper(clean, 0.2)
    for name, fn in [("固定3x3", lambda x: median_fixed(x, 3)),
                     ("固定7x7", lambda x: median_fixed(x, 7)),
                     ("AMF(S_max=7)", lambda x: adaptive_median(x, 7))]:
        out = fn(noisy)
        print(f"  {name:14s} PSNR={psnr(out, clean):6.2f} dB  SSIM={ssim(out, clean):.4f}")
