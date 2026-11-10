import copy
import numpy as np
import torch
import scipy.io as scio
import os
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



# @singleton
class PhysicalModel():
    def __init__(self, cube_size,shift_size,forward_model_path,shift_step=1,rotate_90=-1,crop_mat=True,crop_roi=None,test_cube_size=None):
        self.cube_size = cube_size
        self.shift_size = shift_size
        self.shift_step = shift_step
        self.forward_model_path = forward_model_path
        self.rotate_90 = rotate_90
        self.need_crop_mat = crop_mat
        self.crop_roi = crop_roi
        self.test_cube_size = test_cube_size

        # Core operator initialization
        self.shift_op, self.shift_back_op = self._create_shift_ops(self.cube_size)
        self.A = lambda x, phi: A_torch_withshift(x, phi, self.shift_op, self.shift_back_op)
        self.At = lambda x, phi: At_torch_withshift(x, phi, self.shift_op, self.shift_back_op)

        # Load and process model data
        self._mat = self._load_and_process_mat()
        self.mat = copy.deepcopy(self._mat)
        self._init_clip_mask()
        self.crop_mat(is_first_crop=True)

        # Main model parameters
        self.phi, self.phis = self._get_phi_phis(self.mat['mask_3d_shift'], self.cube_size)
        self.csr = self._get_csr(self.crop_roi_index, self.cube_size).detach()
        self.cube2meas = lambda x: self.A(self.shift_op(x), self.phi)
        # Used during SCALE
        self.csr_cpu = self.csr[0].detach().cpu()
        self.phi_cpu = self.phi[0].detach().cpu()

        # Test-related parameters (initialized on demand)
        self.test_shift_op = None
        self.test_shift_back_op = None
        self.test_A = None
        self.test_At = None
        self.test_phi = None
        self.test_phis = None
        self.test_csr = None
        self.test_csr_cpu = None
        self.test_phi_cpu = None
        self.test_clip_01 = None
        if self.test_cube_size is not None:
            self._init_test_components()

    def _create_shift_ops(self, cube_size):
        """Create shift operators for the specified size"""
        return make_shift_operator(cube_size, self.shift_size, self.shift_step)

    def refresh_phi_csr(self):
        """Refresh phi and csr parameters"""
        self.crop_mat()
        self.phi, self.phis = self._get_phi_phis(self.mat['mask_3d_shift'], self.cube_size)
        self.csr = self._get_csr(self.crop_roi_index, self.cube_size)
        self.csr_cpu = self.csr[0].detach().cpu()
        self.phi_cpu = self.phi[0].detach().cpu()

    def _init_test_components(self):
        """Initialize test-related components"""
        self.test_shift_op, self.test_shift_back_op = self._create_shift_ops(self.test_cube_size)
        self.test_A = lambda x, phi: A_torch_withshift(x, phi, self.test_shift_op, self.test_shift_back_op)
        self.test_At = lambda x, phi: At_torch_withshift(x, phi, self.test_shift_op, self.test_shift_back_op)
        self.test_clip_01 = self._init_test_clip_mask()
        test_mask = self._get_test_phi()
        self.test_phi, self.test_phis = self._get_phi_phis(test_mask, self.test_cube_size)
        self.test_csr = self._get_csr(self.get_test_roi(), self.test_cube_size)
        self.test_csr_cpu = self.test_csr[0].detach().cpu()
        self.test_phi_cpu = self.test_phi[0].detach().cpu()

    def _get_phi_phis(self, mask, cube_size):
        """General method: get phi and phis"""
        coded_aperture_shift = numpy2torch(mask)
        coded_aperture_shift_cuda = coded_aperture_shift.cuda().to(torch.float32)
        phi = coded_aperture_shift_cuda.repeat(cube_size[0], 1, 1, 1)

        phi_s = torch.sum(phi ** 2, 1)
        phi_s[phi_s == 0] = 1
        return phi, phi_s

    def _load_and_process_mat(self):
        """Load and preprocess mat file"""
        mat = scio.loadmat(self.forward_model_path)
        if "mask_3d_shift" not in mat:
            mat["mask_3d_shift"] = mat["Mask"]

        # Handle rotation
        h, w, c = mat["mask_3d_shift"].shape
        if self.rotate_90 != 0:
            rotated_mask = np.zeros((w, h, c))
            for i in range(c):
                rotated_mask[:, :, i] = np.rot90(mat["mask_3d_shift"][:, :, i], k=self.rotate_90)
            mat["mask_3d_shift"] = rotated_mask
        # Handle lens shading
        if 'lsPan' in mat:
            if self.rotate_90 != 0:
                mat['lsPan'] = np.rot90(mat['lsPan'], k=self.rotate_90)
            else:
                mat['lsPan'] = np.rot90(mat['lsPan'])
            mat['useLs'] = True
        else:
            mat['lsPan'] = np.ones(shape=[mat["mask_3d_shift"].shape[0], mat["mask_3d_shift"].shape[1]-self.shift_size,mat["mask_3d_shift"].shape[2]])
            mat['useLs'] = False
        return mat

    def _init_clip_mask(self):
        """Initialize crop mask"""
        b, c, h, w = self.cube_size
        self.clip_01 = np.zeros(shape=(h, w + self.shift_size, c))
        for i in range(c):
            self.clip_01[:, i:w + i, i] = 1
    def _init_test_clip_mask(self):
        """Initialize crop mask"""
        b, c, h, w = self.test_cube_size
        clip_01 = np.zeros(shape=(h, w + self.shift_size, c))
        for i in range(c):
            clip_01[:, i:w + i, i] = 1

        return clip_01

    def crop_mat(self, is_first_crop=False):
        if not self.need_crop_mat:
            return
        total_h, total_w = self._mat['mask_3d_shift'].shape[0], self._mat['mask_3d_shift'].shape[1]
        crop_h, crop_w = self.cube_size[2], self.cube_size[3]
        print(f'mask shape before crop: {self._mat["mask_3d_shift"].shape},{self.crop_roi}')
        # Calculate valid boundaries within the crop_roi range
        x1_min = max(0, self.crop_roi[0])
        x1_max = min(total_w - crop_w - self.shift_size, self.crop_roi[1] - crop_w)
        y1_min = max(0, self.crop_roi[2])
        y1_max = min(total_h - crop_h, self.crop_roi[3] - crop_h)

        # Range validation
        if x1_min > x1_max or y1_min > y1_max:
            raise ValueError(f"Invalid crop range: x[{x1_min}:{x1_max}], y[{y1_min}:{y1_max}]")

        # Randomly select crop starting point
        x1 = np.random.randint(x1_min, x1_max + 1)
        y1 = np.random.randint(y1_min, y1_max + 1)
        x2, y2 = x1 + crop_w, y1 + crop_h
        print(f"Crop mat at: x[{x1}:{x2}], y[{y1}:{y2}]")

        self.mat['mask_3d_shift'] = self._mat['mask_3d_shift'][y1:y2, x1:x2 + self.shift_size] * self.clip_01
        self.mat['lsPan'] = self._mat['lsPan'][y1:y2, x1:x2]
        self.crop_roi_index = (x1, x2, y1, y2)

    def get_test_roi(self):
        """Get test crop region"""
        crop_h, crop_w = self.test_cube_size[2], self.test_cube_size[3]
        x1 = self.crop_roi[0] if self.crop_roi else 0
        y1 = self.crop_roi[2] if self.crop_roi else 0
        return (x1, x1 + crop_w, y1, y1 + crop_h)

    def _get_test_phi(self):
        """Get phi for testing"""
        x1, x2, y1, y2 = self.get_test_roi()
        return self._mat['mask_3d_shift'][y1:y2, x1:x2 + self.shift_size] * self.test_clip_01

    def _get_csr(self, roi, cube_size):

        # Crop to the specified region
        x1, x2, y1, y2 = roi
        ls_pan_cropped = self._mat['lsPan'][y1:y2, x1:x2]

        # Convert to CUDA tensor
        ls_pan_cuda = torch.from_numpy(ls_pan_cropped.copy()).unsqueeze(0).cuda().to(torch.float32)
        ls_pan_cuda = ls_pan_cuda.permute(0,3,1,2)
        ls_pan_cuda = ls_pan_cuda.sum(dim=1, keepdim=True) / 30.0

        csr_old = [0.24821647, 0.25391061, 0.25899804, 0.26646457, 0.27490154,0.28410495, 0.29161091, 0.29716643, 0.30114751, 0.30493644,0.30970117, 0.31743852, 0.32543948, 0.33140972, 0.33762199,0.34857567, 0.35947739, 0.36372839, 0.3691492 , 0.37141317,0.3761853 , 0.39116255, 0.39713657, 0.39823287, 0.39606224,0.39656254, 0.38578051, 0.37696856, 0.35968393, 0.34055519]
        csr_old = np.array(csr_old)
        csr = csr_old

        csr_cuda = torch.from_numpy(csr).reshape(1, -1, 1, 1).cuda().to(torch.float32)
        csr_cuda = csr_cuda * ls_pan_cuda

        return csr_cuda.repeat(cube_size[0], 1, 1, 1)
    def _save_adjust_mat(self):
        mat = scio.loadmat(self.forward_model_path)
        print(mat.keys())
        print(mat['Mask'].shape, mat['cameraSpectralResponse'].shape, mat['Mask'].dtype,
              mat['cameraSpectralResponse'].dtype)
        if "mask_3d_shift" not in mat:
            mat["mask_3d_shift"] = mat["Mask"]

        # Handle rotation
        h, w, c = mat["mask_3d_shift"].shape
        if self.rotate_90 != 0:
            rotated_mask = np.zeros((w, h, c))
            for i in range(c):
                rotated_mask[:, :, i] = np.rot90(mat["mask_3d_shift"][:, :, i], k=self.rotate_90)
            mat["mask_3d_shift"] = rotated_mask
        # 0904 update
        phi_vector = [0.21691034734249115, 0.18890435993671417, 0.1842731535434723, 0.16737677156925201,
                      0.19550390541553497, 0.2301868498325348, 0.2548943758010864, 0.2920522391796112, 0.33815571665763855,
                      0.3624413013458252, 0.38824692368507385, 0.3851827383041382, 0.38759106397628784, 0.38869795203208923,
                      0.4192715585231781, 0.465465247631073, 0.5598520636558533, 0.6117437481880188, 0.6328190565109253,
                      0.7507672309875488, 0.7859675288200378, 0.9031936526298523, 0.9348198175430298, 0.9453418254852295,
                      1.006991982460022, 1.172441005706787, 1.5449787378311157, 1.7285327911376953, 1.3224397897720337,
                      0.5945279598236084]
        csr_vector = [0.24136176705360413, 0.2609690725803375, 0.29118531942367554, 0.32248613238334656, 0.3601146340370178,
                      0.38807517290115356, 0.4061974287033081, 0.39970993995666504, 0.41175922751426697, 0.4159480929374695,
                      0.4268602132797241, 0.439266562461853, 0.46053293347358704, 0.48065873980522156, 0.5189126133918762,
                      0.5563479065895081, 0.5854666829109192, 0.606160044670105, 0.6209211349487305, 0.6630484461784363,
                      0.9108121395111084, 1.1011842489242554, 1.2175161838531494, 1.2912153005599976, 1.3895609378814697,
                      1.518876314163208, 1.709312081336975, 1.9860728979110718, 2.072974920272827, 1.8592581748962402]
        cassi_dx = 0.9529260396957397
        cassi_dy = 0.4113391041755676
        phi_vector = np.array(phi_vector).reshape(1, 1, -1)
        csr_vector = np.array(csr_vector).reshape(1, 1, -1)
        mask_3d_shift = mat["mask_3d_shift"] * phi_vector
        cameraSpectralResponse = mat['cameraSpectralResponse'] * csr_vector.reshape(-1)

        # Apply subpixel shift to phi
        # Import shift_image function for subpixel shifting
        try:
            from run_scripts.infer_scripts.sciexps.DCCHI.fmodel.偏移仿真建模.小测试.生成偏移的卷积核_可微 import shift_image
        except ImportError:
            print("Warning: shift_image function not found, subpixel shifting will be skipped")
            shift_image = None

        if shift_image is not None:
            # Convert to torch.float32 for computation
            mask_shifted = torch.from_numpy(mask_3d_shift).to(torch.float32).cuda()

            # All channels use the same shift, can be processed in parallel
            h, w, c = mask_shifted.shape
            # Rearrange dimensions to fit shift_image function: (1, c, h, w) - batch=1, channels=c
            mask_batch = mask_shifted.permute(2, 0, 1).unsqueeze(0)  # (1, c, h, w)

            # Process all channels in parallel
            shifted_batch = shift_image(mask_batch, cassi_dx, cassi_dy)  # (1, c, h, w)

            # Reshape back to original dimensions: (h, w, c)
            mask_shifted = shifted_batch.squeeze(0).permute(1, 2, 0)  # (h, w, c)
            print(f"mask_shifted dtype after shifting: {mask_shifted.dtype}")
            # Convert back to numpy
            mask_shifted_np = mask_shifted.cpu().numpy()

            # If rotated before, rotate back now
            if self.rotate_90 != 0:
                # Rotate back to the original orientation (reverse rotation)
                original_mask = np.zeros((w, h, c))  # Note the h,w order here
                for i in range(c):
                    original_mask[:, :, i] = np.rot90(mask_shifted_np[:, :, i], k=-self.rotate_90)
                mask_shifted_np = original_mask

            # Save to a new mat file
            output_mat = scio.loadmat(self.forward_model_path)
            output_mat["Mask"] = mask_shifted_np
            output_mat["cameraSpectralResponse"] = cameraSpectralResponse
            output_mat["cassi_dx"] = cassi_dx
            output_mat["cassi_dy"] = cassi_dy
            output_mat["phi_vector"] = phi_vector
            output_mat["csr_vector"] = csr_vector

            # Generate output filename
            base_path = self.forward_model_path.rsplit('.', 1)[0]
            output_path = f"{base_path}.mat"

            try:
                scio.savemat(output_path, output_mat)
                print(f"Successfully saved shifted mask to: {output_path}")
                print(f"Applied subpixel shift: dx={cassi_dx}, dy={cassi_dy}")
            except Exception as e:
                print(f"Error saving mat file: {e}")
        else:
            print("shift_image function not available, skipping subpixel shifting")


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

if __name__ == '__main__':
    from physical.dcchi.dcchi_utils import get_shift, singleton, numpy2torch
    from physical.cassi.cassi_utils import A_torch_withshift, At_torch_withshift, A_torch, At_torch
    import conf

    conf.get_config()
    opt = conf.opt


    cassi_model = PhysicalModel(
        cube_size=(opt.batch_size, opt.data_c, opt.data_h, opt.data_w),
        shift_size=opt.shift_size,
        forward_model_path=opt.forward_model_path,
        shift_step=opt.shift_step,
        rotate_90=opt.rotate_90,
        crop_mat=opt.crop_mat
    )
    cassi_model.crop_mat(True)
    for i in range(1000000):
        print("running")
        cassi_model.crop_mat(is_first_crop=False)
