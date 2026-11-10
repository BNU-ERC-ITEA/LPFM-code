"伪彩色逐通道可视化"
import cv2
import numpy as np
import torch
from einops import rearrange
from scipy import interpolate
import colorsys

from physical import physical_generator
from tools.utils import joinc
import torch

import numpy as np
from scipy.interpolate import PchipInterpolator
from typing import Union, Optional
from visual.visual_per_bands import visual_x2

def spectrum_rgb1(lambda_in: Union[float, np.ndarray],
                 matching_fcn: str = '1964_full') -> np.ndarray:
    """
    Converts a spectral wavelength to RGB.

    Parameters
    ----------
    lambda_in : float or array-like
        Spectral wavelength(s) in nanometers. Must be a scalar or 1D array.
    matching_fcn : str, optional
        Color matching function to use. Default is '1964_full'.
        See color_match_fcn for supported options.

    Returns
    -------
    srgb : ndarray
        RGB values in the sRGB color space.
        Shape depends on input: scalar returns (3,), vector returns (n, 3).

    Raises
    ------
    ValueError
        If input is not a scalar or 1D array.

    Examples
    --------
    # >>> rgb = spectrum_rgb(550)  # Green wavelength
    # >>> rgb = spectrum_rgb([450, 550, 650])  # Blue, green, red

    See Also
    --------
    color_match_fcn, create_spectrum, spectrum_label
    """
    # Convert input to numpy array
    lambda_in = np.asarray(lambda_in)

    # Check if input is scalar or 1D vector
    if lambda_in.ndim > 1:
        raise ValueError('Input must be a scalar or vector of wavelengths.')

    # Ensure it's at least 1D for consistent processing
    is_scalar = lambda_in.ndim == 0
    if is_scalar:
        lambda_in = lambda_in.reshape(1)

    # Get the color matching functions
    lambda_match, x_fcn, y_fcn, z_fcn = color_match_fcn(matching_fcn)

    # Interpolate the input wavelength in the color matching functions
    # Using PCHIP interpolation with extrapolation set to 0
    x_interp = PchipInterpolator(lambda_match, x_fcn, extrapolate=False)
    y_interp = PchipInterpolator(lambda_match, y_fcn, extrapolate=False)
    z_interp = PchipInterpolator(lambda_match, z_fcn, extrapolate=False)

    X = x_interp(lambda_in)
    Y = y_interp(lambda_in)
    Z = z_interp(lambda_in)

    # Replace NaN values (from extrapolation) with 0
    X = np.nan_to_num(X, nan=0.0)
    Y = np.nan_to_num(Y, nan=0.0)
    Z = np.nan_to_num(Z, nan=0.0)

    # Stack XYZ values
    XYZ = np.stack([X, Y, Z], axis=-1)

    # Convert XYZ to sRGB
    srgb = xyz_to_srgb(XYZ)

    # Return scalar result if input was scalar
    if is_scalar:
        return srgb[0]

    return srgb


def xyz_to_srgb(xyz: np.ndarray) -> np.ndarray:
    """
    Convert XYZ color space to sRGB.

    Parameters
    ----------
    xyz : ndarray
        XYZ color values. Last dimension must be 3.

    Returns
    -------
    srgb : ndarray
        sRGB color values in range [0, 1].
    """
    # XYZ to sRGB conversion matrix (D65 illuminant)
    # Based on sRGB standard
    M = np.array([
        [3.2406, -1.5372, -0.4986],
        [-0.9689, 1.8758, 0.0415],
        [0.0557, -0.2040, 1.0570]
    ])

    # Apply transformation
    rgb_linear = np.dot(xyz, M.T)

    # Apply sRGB gamma correction
    srgb = np.where(
        rgb_linear <= 0.0031308,
        12.92 * rgb_linear,
        1.055 * np.power(rgb_linear, 1 / 2.4) - 0.055
    )

    # Clip to valid range [0, 1]
    srgb = np.clip(srgb, 0, 1)

    return srgb


def spectrum_rgb(lambda_in):
    """
    增强版波长到RGB转换，对红外波段也有效
    """
    lambda_in = np.asarray(lambda_in)
    is_scalar = lambda_in.ndim == 0
    if is_scalar:
        lambda_in = lambda_in.reshape(1)

    rgb = np.array([wavelength_to_color(wl) for wl in lambda_in])

    if is_scalar:
        return rgb[0]
    return rgb
def wavelength_to_color(wavelength):
    """
    将波长映射到假彩色（对红外波段也有效）
    """
    # 可见光范围 380-700nm 使用光谱色
    # 近红外 700-1000nm 映射到紫色-品红色

    if wavelength >= 380 and wavelength < 440:
        # 紫色
        R = -(wavelength - 440) / (440 - 380)
        G = 0.0
        B = 1.0
    elif wavelength >= 440 and wavelength < 490:
        # 蓝色
        R = 0.0
        G = (wavelength - 440) / (490 - 440)
        B = 1.0
    elif wavelength >= 490 and wavelength < 510:
        # 青色
        R = 0.0
        G = 1.0
        B = -(wavelength - 510) / (510 - 490)
    elif wavelength >= 510 and wavelength < 580:
        # 绿色到黄色
        R = (wavelength - 510) / (580 - 510)
        G = 1.0
        B = 0.0
    elif wavelength >= 580 and wavelength < 645:
        # 黄色到橙色
        R = 1.0
        G = -(wavelength - 645) / (645 - 580)
        B = 0.0
    elif wavelength >= 645 and wavelength <= 700:
        # 红色
        R = 1.0
        G = 0.0
        B = 0.0
    elif wavelength > 700 and wavelength <= 780:
        # 近红外：深红到品红
        t = (wavelength - 700) / (780 - 700)
        R = 1.0
        G = 0.0
        B = t * 0.8  # 逐渐加入蓝色分量
    # elif wavelength > 700 and wavelength <= 780:
    #     # 近红外：红色平滑过渡到品红
    #     t = (wavelength - 700) / (780 - 700)
    #     R = 1.0
    #     G = 0.0
    #     B = t  # 从0线性增加到1，实现红色→品红的过渡
    else:
        # 超出范围，映射到品红-紫色
        t = min((wavelength - 780) / (1000 - 780), 1.0)
        R = 1.0 - t * 0.3
        G = 0.0
        B = 0.8 + t * 0.2

    # 强度衰减（边缘波段稍暗）
    if wavelength < 420:
        factor = 0.3 + 0.7 * (wavelength - 380) / (420 - 380)
    elif wavelength > 700:
        factor = 0.7 + 0.3 * max(0, (780 - wavelength) / (780 - 700))
    else:
        factor = 1.0

    R *= factor
    G *= factor
    B *= factor

    return np.array([R, G, B])


def color_match_fcn1(matching_fcn: str = '1964_full'):
    """
    Returns CIE color matching functions.

    Parameters
    ----------
    matching_fcn : str
        Type of color matching function. Options:
        - '1931': CIE 1931 2-degree standard observer
        - '1964_full': CIE 1964 10-degree standard observer (full range)

    Returns
    -------
    lambda_match : ndarray
        Wavelengths in nanometers
    x_fcn : ndarray
        X tristimulus values
    y_fcn : ndarray
        Y tristimulus values
    z_fcn : ndarray
        Z tristimulus values
    """
    if matching_fcn == '1964_full':
        # CIE 1964 10-degree standard observer (380-780 nm, 5nm intervals)
        lambda_match = np.arange(380, 785, 5)

        # Simplified example data - you should use actual CIE 1964 data
        # This is placeholder data for demonstration
        x_fcn = np.zeros_like(lambda_match, dtype=float)
        y_fcn = np.zeros_like(lambda_match, dtype=float)
        z_fcn = np.zeros_like(lambda_match, dtype=float)

        # Peak approximations (replace with actual CIE data)
        x_fcn = np.exp(-((lambda_match - 570) ** 2) / (2 * 50 ** 2))
        y_fcn = np.exp(-((lambda_match - 550) ** 2) / (2 * 45 ** 2))
        z_fcn = np.exp(-((lambda_match - 445) ** 2) / (2 * 30 ** 2))

    elif matching_fcn == '1931':
        # CIE 1931 2-degree standard observer
        lambda_match = np.arange(380, 785, 5)

        # Placeholder - use actual CIE 1931 data
        x_fcn = np.zeros_like(lambda_match, dtype=float)
        y_fcn = np.zeros_like(lambda_match, dtype=float)
        z_fcn = np.zeros_like(lambda_match, dtype=float)

    else:
        raise ValueError(f"Unsupported matching function: {matching_fcn}")

    return lambda_match, x_fcn, y_fcn, z_fcn



import numpy as np
import torch
import matplotlib.pyplot as plt
import torch
import numpy as np
import math
def t1(hsi,bands,outf,show=False):




    base_rgbs = []
    for i in range(len(bands)):
        base_rgbs.append(spectrum_rgb(bands[i]))
    base_rgbs = np.array(base_rgbs)
    base_rgbs = torch.tensor(base_rgbs).float().cuda() # c 3
    hsi_rgb = hsi.unsqueeze(2) # b c 1 h w
    base_rgb = base_rgbs.reshape(1, -1, 3, 1, 1) # 1 c 3 1 1
    hsi_rgb = hsi_rgb * base_rgb # b c 3 h w



    # 假设 hsi_rgb: [B, C, 3, H, W]
    b = 0
    hsi_rgb_vis = hsi_rgb[b]  # [C, 3, H, W]
    num_bands = hsi_rgb_vis.shape[0]

    # 转换为 numpy 并归一化到 [0,1]
    hsi_rgb_vis = hsi_rgb_vis.detach().cpu().permute(0, 2, 3, 1).numpy()  # [C, H, W, 3]
    hsi_rgb_vis = (hsi_rgb_vis - hsi_rgb_vis.min()) / (hsi_rgb_vis.max() - hsi_rgb_vis.min() + 1e-8)

    # 计算合适的行列数（接近正方形）
    cols = math.ceil(math.sqrt(num_bands))
    rows = math.ceil(num_bands / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))

    # 保证 axes 可迭代
    axes = np.array(axes).reshape(rows, cols)
    # print(hsi_rgb_vis.shape[])
    for idx in range(hsi_rgb_vis.shape[0]):
        ax = axes.flat[idx]
        if idx < num_bands:
            ax.imshow(hsi_rgb_vis[idx])
            ax.set_title(f"{bands[idx]}nm", fontsize=10)
        ax.axis('off')
        img = (hsi_rgb_vis[idx] * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(joinc(outf,f"{bands[idx]}nm.png"), img)
        print(f"save {joinc(outf,f'{bands[idx]}nm.png')}")
    plt.tight_layout()


    layout_save_path = joinc(outf,"all_bands_layout.png")
    plt.savefig(
        layout_save_path,
        dpi=300,  # 分辨率（越高越清晰，默认100）
        bbox_inches='tight',  # 自动裁剪空白区域（避免子图被截断）
        pad_inches=0.1  # 图片边缘留白（可选，默认0.1）
    )
    print(f"保存整个子图布局到：{layout_save_path}")
    plt.show()

def t2(hsi,bands,outf):




    base_rgbs = []
    for i in range(len(bands)):
        base_rgbs.append(spectrum_rgb(bands[i]))
    base_rgbs = np.array(base_rgbs)
    base_rgbs = torch.tensor(base_rgbs).float().cuda() # c 3
    hsi_rgb = hsi.unsqueeze(2) # b c 1 h w
    base_rgb = base_rgbs.reshape(1, -1, 3, 1, 1) # 1 c 3 1 1
    hsi_rgb = hsi_rgb * base_rgb # b c 3 h w



    # 假设 hsi_rgb: [B, C, 3, H, W]
    b = 0
    hsi_rgb_vis = hsi_rgb[b]  # [C, 3, H, W]
    num_bands = hsi_rgb_vis.shape[0]

    # 转换为 numpy 并归一化到 [0,1]
    hsi_rgb_vis = hsi_rgb_vis.detach().cpu().permute(0, 2, 3, 1).numpy()  # [C, H, W, 3]
    # hsi_rgb_vis = hsi_rgb_vis  / hsi_rgb_vis.max()

    # 计算合适的行列数（接近正方形）
    # cols = math.ceil(math.sqrt(num_bands))
    # rows = math.ceil(num_bands / cols)
    #
    # fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    #
    # # 保证 axes 可迭代
    # axes = np.array(axes).reshape(rows, cols)
    # # print(hsi_rgb_vis.shape[])
    imgs = []
    for idx in range(hsi_rgb_vis.shape[0]):
        # ax = axes.flat[idx]
        # if idx < num_bands:
        #     ax.imshow(hsi_rgb_vis[idx])
        #     ax.set_title(f"{bands[idx]}nm", fontsize=10)
        # ax.axis('off')
        img = (hsi_rgb_vis[idx] * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        imgs.append(img)
        if outf is not None:
            cv2.imwrite(joinc(outf,f"{bands[idx]}nm.png"), img)
            print(f"save {joinc(outf,f'{bands[idx]}nm.png')}")
    return imgs
def ga(image, gamma=0.75):
    """伽马校正"""
    img = image.copy()
    if img.dtype == np.uint8:
        img = img.astype(np.float64) / 255.0
    img = np.power(np.clip(img, 1e-8, 1.0), gamma)
    if image.dtype == np.uint8:
        img = (img * 255).astype(np.uint8)
    return img
def t2_with_gamma(hsi, bands, outf,gamma = 0.75):

    base_rgbs = []
    for i in range(len(bands)):
        base_rgbs.append(spectrum_rgb(bands[i]))
    base_rgbs = np.array(base_rgbs)
    base_rgbs = torch.tensor(base_rgbs).float().cuda()  # c 3
    hsi_rgb = hsi.unsqueeze(2)  # b c 1 h w
    base_rgb = base_rgbs.reshape(1, -1, 3, 1, 1)  # 1 c 3 1 1
    hsi_rgb = hsi_rgb * base_rgb  # b c 3 h w

    # 假设 hsi_rgb: [B, C, 3, H, W]
    b = 0
    hsi_rgb_vis = hsi_rgb[b]  # [C, 3, H, W]
    num_bands = hsi_rgb_vis.shape[0]

    # 转换为 numpy 并归一化到 [0,1]
    hsi_rgb_vis = hsi_rgb_vis.detach().cpu().permute(0, 2, 3, 1).numpy()  # [C, H, W, 3]
    # hsi_rgb_vis = hsi_rgb_vis  / hsi_rgb_vis.max()

    # 计算合适的行列数（接近正方形）
    # cols = math.ceil(math.sqrt(num_bands))
    # rows = math.ceil(num_bands / cols)
    #
    # fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    #
    # # 保证 axes 可迭代
    # axes = np.array(axes).reshape(rows, cols)
    # # print(hsi_rgb_vis.shape[])

    imgs = []
    for idx in range(hsi_rgb_vis.shape[0]):
        # ax = axes.flat[idx]
        # if idx < num_bands:
        #     ax.imshow(hsi_rgb_vis[idx])
        #     ax.set_title(f"{bands[idx]}nm", fontsize=10)
        # ax.axis('off')
        img = (hsi_rgb_vis[idx] * 255).astype(np.uint8)
        img = ga(img, gamma)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        imgs.append(img)
        if outf is not None:
            cv2.imwrite(joinc(outf, f"{bands[idx]}nm.png"), img)
            print(f"save {joinc(outf, f'{bands[idx]}nm.png')}")
    return imgs
    # plt.tight_layout()
def t2_gamma_alpha(hsi, bands, outf, gamma: Optional[float] = None):

    base_rgbs = []
    for i in range(len(bands)):
        base_rgbs.append(spectrum_rgb(bands[i]))
    base_rgbs = np.array(base_rgbs)
    base_rgbs = torch.tensor(base_rgbs).float().cuda()  # c 3
    hsi_rgb = hsi.unsqueeze(2)  # b c 1 h w
    base_rgb = base_rgbs.reshape(1, -1, 3, 1, 1)  # 1 c 3 1 1
    hsi_rgb = hsi_rgb * base_rgb  # b c 3 h w

    # 假设 hsi_rgb: [B, C, 3, H, W]
    b = 0
    hsi_rgb_vis = hsi_rgb[b]  # [C, 3, H, W]
    num_bands = hsi_rgb_vis.shape[0]

    # 转换为 numpy 并归一化到 [0,1]
    hsi_rgb_vis = hsi_rgb_vis.detach().cpu().permute(0, 2, 3, 1).numpy()  # [C, H, W, 3]
    # hsi_rgb_vis = hsi_rgb_vis  / hsi_rgb_vis.max()

    # 计算合适的行列数（接近正方形）
    # cols = math.ceil(math.sqrt(num_bands))
    # rows = math.ceil(num_bands / cols)
    #
    # fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    #
    # # 保证 axes 可迭代
    # axes = np.array(axes).reshape(rows, cols)
    # # print(hsi_rgb_vis.shape[])
    imgs = []
    for idx in range(hsi_rgb_vis.shape[0]):

        # ---- 原来的 gamma 校正部分 ----
        if gamma is not None:
            hsi_rgb_vis[idx] = gamma_correction(hsi_rgb_vis[idx], gamma=gamma)
            hsi_rgb_vis[idx] = np.clip(hsi_rgb_vis[idx], 0, 1)

        # 伪彩色图（当前波段）
        pseudo_rgb = hsi_rgb_vis[idx]  # [H, W, 3]  float [0,1]
        pseudo_rgb_uint8 = (pseudo_rgb * 255).astype(np.uint8)

        # 取原始HSI单波段
        band_gray = hsi[b, idx].detach().cpu().numpy()  # [H, W]

        band_gray = np.clip(band_gray, 0, 1)
        band_gray_uint8 = (band_gray * 255).astype(np.uint8)

        # 转成3通道灰度图
        band_gray_3c = cv2.cvtColor(band_gray_uint8, cv2.COLOR_GRAY2RGB)

        # ---- 关键：50%透明度叠加 ----
        blended = cv2.addWeighted(
            pseudo_rgb_uint8, 0.9,  # 伪彩色图 50%
            band_gray_3c, 0.1,  # 原始波段 50%
            0
        )

        # 转回OpenCV的BGR格式用于保存
        blended = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)

        if outf is not None:
            cv2.imwrite(joinc(outf, f"{bands[idx]}nm_overlay.png"), blended)
            print(f"save {joinc(outf, f'{bands[idx]}nm_overlay.png')}")

        imgs.append(blended)
    return imgs


def gamma_correction(img, gamma):
    # 归一化到0-1
    img_norm = img
    # gamma变换
    img_gamma = np.power(img_norm, gamma)

    return img_gamma
