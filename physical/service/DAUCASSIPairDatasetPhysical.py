import numpy as np
import torch

from physical.cassi.cassi_physical_model_pair_dataset import PhysicalModel as CASSIPhysicalModel

class DAUCASSIPhysical():
    def __init__(self, opt):
        self.opt = opt
        if hasattr(opt,"crop_roi"):
            crop_roi = opt.crop_roi
        else:
            crop_roi = None
        self.cassi_model = CASSIPhysicalModel(
            cube_size=(opt.batch_size, opt.data_c, opt.data_h, opt.data_w),
            shift_size=opt.shift_size,
            forward_model_path=opt.forward_model_path,
            shift_step=opt.shift_step,
            rotate_90=opt.rotate_90,
            crop_mat= opt.crop_mat,
            crop_roi=crop_roi,
            test_cube_size=(opt.batch_size,opt.data_c, opt.test_data_h, opt.data_w),
        )


    # This part is not used!!!
    def forward(self,batch,stage="train"):
        if stage == "train":

            image = batch
            phi = self.cassi_model.phi
            input_meas = self.cassi_model.A(self.cassi_model.shift_op(image), phi)
            gt = image
        elif stage == "test":
            image = batch
            phi = self.cassi_model.test_phi
            input_meas = self.cassi_model.test_A(self.cassi_model.test_shift_op(image), phi)
            gt = image
        else:
            raise Exception("forward stage error")
        return gt,input_meas

class DAUCASSIPhysicalMode1(DAUCASSIPhysical):
    # Mode 1 training
    def forward(self,batch,stage="train"):
        if stage == "train":

            image = batch
            phi = self.cassi_model.phi
            input_meas = self.cassi_model.A(self.cassi_model.shift_op(image), phi)
            gt = image
        else:
            raise Exception("forward stage error")
        return gt,input_meas
