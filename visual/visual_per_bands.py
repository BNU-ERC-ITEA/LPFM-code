import cv2
import os
from pathlib import Path
import numpy as np
import scipy.io as sio

join = os.path.join


def write_per_band(img,dst_folder,light_ratio = 2.5):
    os.makedirs(dst_folder, exist_ok=True)
    mat = img.copy()
    mat_ = mat[mat > 0.05]
    mat = mat / (np.mean(mat_) * light_ratio)
    mat = mat

    for i, row in enumerate(range(mat.shape[-1])):
        # lam = int(row[1])
        img = mat[:, :, i]
        img = img * 255
        print(np.max(img))
        img[img > 255] = 255
        img = img.astype(np.int)
        cv2.imwrite(join(dst_folder, f'{row}.bmp'), img)


import os

import cv2
import os
from pathlib import Path
import numpy as np
import scipy.io as sio
join = os.path.join
import torch

import math
def lambda2rgb(wavelength,Gamma=1.00,IntensityMax=255):
    """
    reference to https://academo.org/demos/wavelength-to-colour-relationship/
    """
    factor, red, green, blue = 0, 0, 0, 0
    if ((wavelength >= 380) and (wavelength < 440)):
        red = -(wavelength - 440) / (440 - 380)
        green = 0.0
        blue = 1.0
    elif ((wavelength >= 440) and (wavelength < 490)):
        red = 0.0
        green = (wavelength - 440) / (490 - 440)
        blue = 1.0
    elif ((wavelength >= 490) and (wavelength < 510)):
        red = 0.0
        green = 1.0
        blue = -(wavelength - 510) / (510 - 490)
    elif ((wavelength >= 510) and (wavelength < 580)):
        red = (wavelength - 510) / (580 - 510)
        green = 1.0
        blue = 0.0
    elif ((wavelength >= 580) and (wavelength < 645)):
        red = 1.0
        green = -(wavelength - 645) / (645 - 580)
        blue = 0.0
    elif ((wavelength >= 645) and (wavelength < 781)):
        red = 1.0
        green = 0.0
        blue = 0.0
    else:
        red = 0.0
        green = 0.0
        blue = 0.0

    if ((wavelength >= 380) and (wavelength < 420)):
        factor = 0.3 + 0.7 * (wavelength - 380) / (420 - 380)
    elif ((wavelength >= 420) and (wavelength < 701)):
        factor = 1.0
    elif ((wavelength >= 701) and (wavelength < 781)):
        factor = 0.3 + 0.7 * (780 - wavelength) / (780 - 700)
    else:
        factor = 0.0

    if (red != 0):
        red = round(IntensityMax * math.pow(red * factor, Gamma))

    if (green != 0):
        green = round(IntensityMax * math.pow(green * factor, Gamma))
    if (blue != 0):
        blue = round(IntensityMax * math.pow(blue * factor, Gamma))
    return [red, green, blue]

def cube_insert(cube, src_bands, dst_bands):
    """
    Interpolate spectral image

    Parameters:
    cube: 3D tensor with shape (c, h, w), representing the spectral image
    src_bands: 1D tensor with shape (num_src_bands), representing the source bands
    dst_bands: 1D tensor with shape (num_dst_bands), representing the target bands

    Returns:
    interp_cube: 3D tensor with shape (num_dst_bands, h, w), representing the interpolated spectral image
    """

    # Ensure input is a torch.Tensor
    if not isinstance(cube, torch.Tensor):
        raise TypeError("cube must be a torch.Tensor")
    if not isinstance(src_bands, torch.Tensor):
        raise TypeError("src_bands must be a torch.Tensor")
    if not isinstance(dst_bands, torch.Tensor):
        raise TypeError("dst_bands must be a torch.Tensor")

    # Get cube dimensions
    c, h, w = cube.size()

    # Ensure src_bands size matches the spectral dimension of the cube
    assert c == src_bands.size(0), f"src_bands size {src_bands.size(0)} must match the spectral dimension {c} of the cube"

    # Interpolate along the spectral dimension
    indices = torch.searchsorted(src_bands, dst_bands)  # Find the position of target bands in source bands
    low_index = torch.clamp(indices - 1, 0, len(src_bands) - 1)  # Low index
    high_index = torch.clamp(indices, 0, len(src_bands) - 1)  # High index

    low_value = cube[low_index, :, :]
    high_value = cube[high_index, :, :]
    interp_weights = (dst_bands - src_bands[low_index]) / (src_bands[high_index] - src_bands[low_index])
    interp_weights[torch.isnan(interp_weights)] = 1.0
    interp_weights = interp_weights.reshape(-1,1,1)
    interp_cube = low_value + interp_weights * (high_value - low_value)
    return interp_cube
def cube2bgr_equalsample(x,bands):
    """
    :param x:
    :param bands:
    :return:
    Uniform sampling for final visualization
    First interpolate x to 10nm uniform spectral bands to ensure sampling consistency
    """
    x_tensor = torch.from_numpy(np.transpose(x,(2,0,1)))
    bands_tensor = torch.from_numpy(bands)
    interval = 10
    min_bands = torch.min(bands_tensor) + 1
    max_bands = (torch.max(bands_tensor) - min_bands)//interval * interval + min_bands
    min_bands,max_bands = int(min_bands),int(max_bands)
    target_bands = torch.Tensor([i for i in range(min_bands,max_bands+interval,interval)])
    x = cube_insert(x_tensor,src_bands=bands_tensor,dst_bands=target_bands).numpy()
    x = np.transpose(x,(1,2,0))
    print(x.shape)
    color_map = get_color_map(target_bands.numpy().reshape(-1))
    bgr = hyper2bgr(x, color_map)
    return bgr


def get_color_map(wl):
    bgrs = []
    for lam in wl:
        r, g, b = lambda2rgb(lam)
        bgrs.append([b, g, r])
    color_map = np.array(np.array(bgrs)) #n*3
    # Normalize columns for white balance
    for i in range(3):
        color_map[:,i] = color_map[:,i]/np.sum(color_map[:,i])
    return  color_map # n*3

def hyper2bgr(hsi_img, color_map):
    h, w, c = hsi_img.shape
    color_map = color_map.T
    bgr = np.zeros((h, w, 3))

    for channel in range(c):

        max_channel = np.max(hsi_img[:, :, channel])
        if max_channel < 1e-6 or np.isnan(max_channel):
            pass
        else:
            hsi_img[:, :, channel] = hsi_img[:, :, channel] / max_channel

    for channel in range(c):
        for i in range(3):
            bgr[:, :, i] = bgr[:, :, i] + color_map[i, channel] * hsi_img[:, :, channel]
    ch = np.sum(color_map, axis=1)
    for i in range(3):
        tt = ch[i]
        bgr[:, :, i] = np.multiply(bgr[:, :, i], 1 / tt)
    bgr = (bgr * 255).astype(np.int32)
    bgr = bgr.astype(np.uint8)
    return bgr


import torch

from einops import einsum
def hyper2bgr_cuda(hsi_img, color_map):
    """
    Convert HSI image to BGR image (PyTorch CUDA version)

    Parameters:
        hsi_img: Input HSI image (H, W, C) torch.Tensor, device is CUDA
        color_map: Color mapping matrix (3, C) torch.Tensor, device is CUDA

    Returns:
        BGR image (H, W, 3) torch.Tensor, device is CUDA
    """
    # Ensure input is on GPU
    if not hsi_img.is_cuda:
        hsi_img = hsi_img.cuda()
    if not color_map.is_cuda:
        color_map = color_map.cuda()
    bgr = einsum(hsi_img, color_map, 'h w c, c c2 -> h w c2')
    bgr = bgr/bgr.max()
    bgr = (bgr * 255).clamp(0, 255).to(torch.uint8)
    return bgr
import matplotlib.pyplot as plt
def adjust_gamma(img, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    new_img = np.empty(img.shape)
    # for i in range(imgs.shape[0]):
    new_img = cv2.LUT(np.array(img, dtype = np.uint8), table)
    return new_img
def dispCube(x,wl,gamma=1.0,save_file=None):
    imh,imw,imc = x.shape


    w = math.floor(math.sqrt(len(wl)))
    h = int(len(wl)//w)

    if w*h < len(wl):
        h = h + 1
    fig, axs = plt.subplots(h, w, figsize=(w * 3, h * 3))
    for i in range(x.shape[-1]):
        ax = axs[int(i//w),int(i%w)]
        img = x[:,:,i]
        ax.imshow(adjust_gamma(img*255,gamma), cmap='gray',vmin=0, vmax=255)
        ax.set_title(f'{wl[i]}nm')
        ax.axis('off')  # Hide axis
    for i in range(len(wl), h * w):
        axs[int(i//w),int(i%w)].axis('off')
    plt.tight_layout()
    if save_file is None:
        plt.show()
    else:
        plt.title(os.path.basename(save_file)[:-4])
        plt.savefig(save_file)
def dispCube2(x,bgr,wl,gamme=1.0,save_file=None):
    dispCube(x,wl,gamme,"tmp.png")
    img2 = cv2.imread("tmp.png")
    os.remove("tmp.png")
    h2,w2,c2 = img2.shape
    h1,w1,c1 = x.shape
    new_h = h2
    new_w = int((h2/h1)*w1)
    bgr = cv2.resize(bgr,(new_w,new_h))
    new_img = cv2.hconcat((bgr,img2))
    if save_file is None:
        plt.close()
        plt.imshow(cv2.cvtColor(new_img,cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    else:

        new_img_ = np.zeros((new_img.shape[0]+190,new_img.shape[1],3))
        new_img_ = new_img_ + 255
        new_img_[:new_img.shape[0],:new_img.shape[1],:] = new_img

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 4
        font_color = (0, 0, 0)  # Black
        line_type = 2
        cv2.putText(
            new_img_,
            os.path.basename(save_file)[:-4],
            (0, new_img.shape[0]+80),
            font,
            font_scale,
            font_color,
            line_type
        )
        cv2.imwrite(save_file, new_img_)

def visual_y(y, file_name):
    """
    Visualize single-channel grayscale image (C dimension is 1)

    Parameters:
        y: Input tensor with shape (1, H, W) or (H, W)
        file_name: Save file name
    """
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    # Convert to numpy array
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()

    # Handle dimensions
    if len(y.shape) == 3 and y.shape[0] == 1:
        y = y[0]  # Remove channel dimension (1, H, W) -> (H, W)

    # Normalize to 0-1 range
    y = y / np.max(y)
    y[y < 0] = 0

    # Convert to 0-255 uint8 image
    y_img = (y * 255).astype(np.uint8)

    # Save grayscale image
    cv2.imwrite(file_name, y_img)
    print(f"Saved grayscale image: {file_name}")


def visual_x2(x,wl,file_name):
    os.makedirs(os.path.dirname(file_name),exist_ok=True)

    x = x / np.max(x)
    x[x < 0] = 0
    bgr = cube2bgr_equalsample(x, wl)
    cv2.imwrite(file_name,bgr)
    #dispCube(x,wl,save_file=join(os.path.dirname(file_name),"lams"+os.path.basename(file_name)))


if __name__ == '__main__':
    wl = np.array([454., 459., 464., 468., 473., 478., 483., 489., 495., 501., 507.,
                   513., 519., 526., 533., 541., 550., 559., 568., 577., 587., 598.,
                   610., 623., 637., 652., 668., 685., 706., 728., 752., 780., 818.,
                   859.])
    src_folder = "/data/wangxin/wangxin/MST_XX/simulation/train_code/exp/p205_3_28/"
    dst_folder = join(src_folder,"bgrlams2")
    os.makedirs(dst_folder,exist_ok=True)
    files = [str(p) for p in Path(src_folder).glob("328_infer.yamlScene3024*.mat")]
    for file in files:
        mat = sio.loadmat(file)
        x = mat['model_out'][0]
        x = x/np.max(x)
        x[x<0] = 0
        bgr = cube2bgr_equalsample(x,wl)
        dispCube2(
            x,
            bgr,
            wl,
            save_file=join(
                dst_folder,
                "lam"+os.path.basename(file).replace(".mat",'.png')
            )
        )
        plt.close()



