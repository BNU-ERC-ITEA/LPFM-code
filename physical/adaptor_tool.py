import torch

from physical.cassi.cassi_physical_model import make_shift_operator, make_shift_operator_cpu


class ShiftOperatorFactory:
    def __init__(self):
        self.shift_ops = {}  # Cache existing shift operations
        self.shift_back_ops = {}  # Cache existing shift_back operations
        self.two_ops = {}

    def create_shift_ops(self, cube_size, shift_size, shift_step):

        shift, shift_back = make_shift_operator(cube_size, shift_size, shift_step)
        return shift, shift_back

    def get_shift(self, shift_config,key="default"):

        if key not in self.shift_ops:
            shift_op, _ = self.create_shift_ops(shift_config["cube_size"], shift_config["shift_size"], shift_config["shift_step"])
            self.shift_ops[key] = shift_op
        return self.shift_ops[key]

    def get_shift_cpu(self, shift_config,key="default"):

        if key not in self.shift_ops:
            shift_op, _ = make_shift_operator_cpu(shift_config["cube_size"], shift_config["shift_size"], shift_config["shift_step"])
            self.shift_ops[key] = shift_op
        return self.shift_ops[key]

    def get_shift_back(self, shift_config,key="default"):

        if key not in self.shift_back_ops:
            _, shift_back_op = self.create_shift_ops(shift_config["cube_size"], shift_config["shift_size"], shift_config["shift_step"])
            self.shift_back_ops[key] = shift_back_op
        return self.shift_back_ops[key]
    def get_two_operator(self,shift_config, key="default"):
        if key not in self.two_ops:
            shift, shift_back = self.create_shift_ops(shift_config["cube_size"], shift_config["shift_size"], shift_config["shift_step"])
            self.two_ops[key] = (shift, shift_back)
        else:
            shift, shift_back = self.two_ops[key]
        return shift, shift_back

    def clear_all(self):
        self.shift_ops.clear()
        self.shift_back_ops.clear()
        self.two_ops.clear()
        print("[ShiftOperatorFactory] All caches cleared.")

shift_factory = ShiftOperatorFactory()


def pad_cassi_data(data, pad_h, pad_w,pad_key):
    """
    Pad the data
    :param data: Input data dictionary
    :param pad_h: Padding size in the height direction
    :param pad_w: Padding size in the width direction
    :return: Padded data dictionary
    """
    skip_key = ['Phi_s']
    for key in data:
        if key in pad_key and not key in skip_key:
            data[key] = torch.nn.functional.pad(data[key], (0, pad_w, 0, pad_h), mode='constant', value=0)
    for key in data:
        if key in skip_key:
            if key == 'Phi_s':
                Phi_s = torch.sum(data['Phi'] ** 2, 1)
                Phi_s[Phi_s == 0] = 1
                data[key] = Phi_s

    return data


def prepare_input_data(y,Phi,shift_config,input_keys,key_meta=""):
    data = {}
    for key in input_keys:
        if key == 'Y':
            data[key] = y
        elif key in ['H', 'SHIFT_H']:
            shift,shift_back = shift_factory.get_two_operator(shift_config, key_meta+"prepare_input_meas")
            y_expanded = y.unsqueeze(1).repeat(1, Phi.shape[1], 1, 1)
            y_expanded = y_expanded / Phi.shape[1] * 2
            H = shift_back(y_expanded)
            if key == 'H':
                data[key] = H
            else:
                SHIFT_H = shift(H)
                data[key] = SHIFT_H  # Use the already computed value directly
        elif key == 'Phi':
            data[key] = Phi
        elif key == 'Phi_s':
            Phi_s = torch.sum(Phi ** 2, 1)
            Phi_s[Phi_s == 0] = 1
            data[key] = Phi_s
        elif key == 'MASK':
            shift_back1 = shift_factory.get_shift_back(shift_config, key_meta+"prepare_input_phi")
            data[key] = shift_back1(Phi)
        else:
            raise ValueError("Invalid input key")
    return data


class PAD_MODE:
    N = 0
    Y = 1
    H = 2
    SHIFT_H = 3


input_key = ['Y','H','UNSHIFT_H','Phi','Phi_s','MASK']
need_pad_key = ['Y','H','UNSHIFT_H','Phi','Phi']


def get_pad(cube_shape, cube_shift_shape, pad_mode, pad_mod):
    if pad_mode == PAD_MODE.SHIFT_H or pad_mode == PAD_MODE.Y:
        src_shape = cube_shift_shape
    elif pad_mode == PAD_MODE.H :
        src_shape = cube_shape
    elif pad_mode == PAD_MODE.N:
        return cube_shape, cube_shift_shape, 0, 0
    else:
        raise ValueError("Invalid pad mode")
    h_mod = src_shape[2] % pad_mod
    w_mod = src_shape[3] % pad_mod
    pad_h = (pad_mod - h_mod) if h_mod != 0 else 0
    pad_w = (pad_mod - w_mod) if w_mod != 0 else 0
    pad_cube_shape = [cube_shape[0], cube_shape[1], cube_shape[2] + pad_h, cube_shape[3] + pad_w]
    pad_cube_shift_shape = [cube_shift_shape[0], cube_shift_shape[1], cube_shift_shape[2] + pad_h, cube_shift_shape[3] + pad_w]
    return pad_cube_shape,pad_cube_shift_shape, pad_h, pad_w

# Single-channel preparation

# Organize the inputs,
# For single channel
# From the observation perspective, inputs are divided into Y, H, unshift(H)
# Forward model inputs are divided into Mask, Phi, Phi_s
# Padding requirements: Y satisfied, H satisfied, unshift(H) satisfied, fewer requirements for Y


