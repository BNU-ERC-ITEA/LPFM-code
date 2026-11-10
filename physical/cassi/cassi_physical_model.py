import copy

import numpy as np
import torch
import scipy.io as scio
from ..utils import get_shift, singleton, numpy2torch
from .cassi_utils import A_torch_withshift, At_torch_withshift, A_torch, At_torch

_, _, shift_torch, shift_back_torch = get_shift()


def make_shift_operator(cube_size,shift_size,step):
    b,c,h,w = cube_size
    shift_back_torch_variable = torch.zeros(b,c,h,w).cuda().to(torch.float32)
    shift_torch_variable = torch.zeros(b,c,h,w + shift_size).cuda().to(torch.float32)
    shift_torch_index = torch.ones_like(shift_back_torch_variable)
    shift_torch_index = shift_torch(shift_torch_index, step=step)
    shift_torch_index = torch.where(shift_torch_index == 1)
    _shift, _shift_back = lambda x: shift_torch(x, step=step, index_tensor=shift_torch_index,
                                                output=shift_torch_variable), \
        lambda x: shift_back_torch(x, step=step, index_tensor=shift_torch_index, output=shift_back_torch_variable)
    return _shift, _shift_back

def make_shift_operator_cpu(cube_size,shift_size,step):
    b,c,h,w = cube_size
    shift_back_torch_variable = torch.zeros(b,c,h,w).to(torch.float32)
    shift_torch_variable = torch.zeros(b,c,h,w + shift_size).to(torch.float32)
    shift_torch_index = torch.ones_like(shift_back_torch_variable)
    shift_torch_index = shift_torch(shift_torch_index, step=step)
    shift_torch_index = torch.where(shift_torch_index == 1)
    _shift, _shift_back = lambda x: shift_torch(x, step=step, index_tensor=shift_torch_index,
                                                output=shift_torch_variable), \
        lambda x: shift_back_torch(x, step=step, index_tensor=shift_torch_index, output=shift_back_torch_variable)
    return _shift, _shift_back


@singleton
class PhysicalModel():
    def __init__(self, cube_size,shift_size,forward_model_path,shift_step=1,rotate_90=-1,crop_mat=True,crop_roi=None):
        self.cube_size = cube_size
        b,c,h,w = cube_size
        self.shift_size = shift_size
        self.shift_step = shift_step
        self.shift_op, self.shift_back_op = self.create_shift_ops()
        self.forward_model_path = forward_model_path
        self.rotate_90 = rotate_90
        self.need_crop_mat = crop_mat
        self.crop_roi = crop_roi

        # Shifted cube, used as input and output, for generating paired data
        self.A = lambda x, phi: A_torch_withshift(x, phi, self.shift_op, self.shift_back_op)
        self.At = lambda x, phi: At_torch_withshift(x, phi, self.shift_op, self.shift_back_op)


        self._mat = self.load_mat()
        self.mat = copy.deepcopy(self._mat)
        clip_01 = np.zeros(shape=(h, w + shift_size, c))
        for i in range(c):
            clip_01[:, i:w + i, i] = 1
        self.clip_01 = clip_01

        self.crop_mat(is_first_crop=True)
        self.phi,self.phis = self.get_phi_phis()

        self.cube2meas = lambda x: self.A(self.shift_op(x), self.phi)

    # The two create operators below are for network calls, not for synthetic data generation
    def create_shift_ops(self):
        return make_shift_operator(self.cube_size, self.shift_size, self.shift_step)
    def create_unshift_ops(self):
        _shift, _shift_back = self.create_shift_ops()
        dual_A = lambda x, Phi, csr: A_torch(x, Phi, _shift, _shift_back)
        dual_At = lambda x, Phi, csr: At_torch(x, Phi, self.cube_size[3] + self.shift_size, _shift, _shift_back)
        return dual_A, dual_At

    def refresh_phi_csr(self):
        self.crop_mat()
        self.phi, self.phis = self.get_phi_phis()

    def get_phi_phis(self):
        mask = self.mat["mask_3d_shift"]
        coded_aperture_shift = numpy2torch(mask)
        coded_aperture_shift_cuda = coded_aperture_shift.cuda().to(torch.float32)
        phi = coded_aperture_shift_cuda.repeat(self.cube_size[0], 1, 1, 1)

        phi_s = torch.sum(phi ** 2, 1)
        phi_s[phi_s == 0] = 1
        return phi,phi_s

    def get_shift_back_phi(self):
        return self.shift_back_op(self.phi)  # Directly apply shift_back to the mask and concatenate to align everything

    def load_mat(self):
        mat = scio.loadmat(self.forward_model_path)
        # Treat the first two dimensions as an image, and rotate the image 90 degrees to the right for all third dimensions
        if "mask_3d_shift" not in mat:
            mat["mask_3d_shift"] = mat["Mask"]
        h, w, c = mat["mask_3d_shift"].shape
        if self.rotate_90!=0:
            Rotated_Mask = np.zeros((w, h, c))
            for i in range(mat["mask_3d_shift"].shape[2]):
                image = mat["mask_3d_shift"][:, :, i]
                rotated_image = np.rot90(image, k=self.rotate_90)  # Rotate the image 90 degrees to the right,-1
                Rotated_Mask[:, :, i] = rotated_image
            mat["mask_3d_shift"] = Rotated_Mask
        return mat


    def crop_mat(self,is_first_crop=False):
        if not self.need_crop_mat:
            return
        total_h, total_w = self._mat['mask_3d_shift'].shape[0], self._mat['mask_3d_shift'].shape[1] - self.shift_size
        crop_h, crop_w = self.cube_size[2], self.cube_size[3]

        if is_first_crop:
            if self.crop_roi is None:
                # Default to cropping from the center on the first crop
                x_center = total_w // 2
                y_center = total_h // 2
                x1 = max(0, x_center - crop_w // 2)
                y1 = max(0, y_center - crop_h // 2)
                x2 = min(total_w, x1 + crop_w)
                y2 = min(total_h, y1 + crop_h)
            else:
                # roi is the roi of the mask region
                x1, x2, y1, y2 = self.crop_roi
                x2 = x2 - self.shift_size
        else:
            # random crop x1,x2,y1,y2
            x1 = np.random.randint(0, total_w - crop_w - self.shift_size + 1)
            y1 = np.random.randint(0, total_h - crop_h + 1)
            x2 = x1 + crop_w
            y2 = y1 + crop_h
        self.mat['mask_3d_shift'] = self._mat['mask_3d_shift'][y1:y2, x1:x2 + self.shift_size] * self.clip_01

@singleton
class PhysicalNoiseModel():
    def __init__(self,phi_gaussian_noise,y_gaussian_noise,y_shot_noise):
        self.phi_gaussian_noise = phi_gaussian_noise
        self.y_gaussian_noise = y_gaussian_noise
        self.y_shot_noise = y_shot_noise
    def forward_param_add_noise(self,phi):
        phi_noise = torch.randn(phi.shape).cuda() * self.phi_gaussian_noise
        phi_ = phi + phi_noise
        phi_[phi_ < 0] = 0

        phi_s = torch.sum(phi_ ** 2, 1)
        phi_s[phi_s == 0] = 1
        return phi, phi_s
    def observation_add_noise(self,y):
        # TODO addd y_noise
        y_noise = torch.randn(y.shape).cuda() * self.y_gaussian_noise
        y_shot_noise = 0
        y_ = y + y_noise + y_shot_noise
        y_[y_< 0] = 0
        return y_
