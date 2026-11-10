import torch
import numpy as np


####Shift mode operators#####
def A_DualCamera_torch_withshift(x, Phi, camera_response_curve, shift_func, shift_back_func):
    """
    Forward operator, both x and Phi have been shifted
    :param x:
    :param Phi:
    :param camera_response_curve: [1,c,1,1]
    :param shift_back_func: Reverse shift operator
    :return:
    """
    y1 = torch.sum(x * Phi, dim=1)  # cassi_measurement
    x_response = shift_back_func(x) * camera_response_curve
    y2 = torch.sum(x_response, dim=1)  # pan_measurement
    if y1.shape[2] == y2.shape[2]:
        # Vertical shift
        y = torch.cat([y1, y2], dim=1)  # concat on vertical
    else:
        y = torch.cat([y1, y2], dim=2)  # concat on vertical
    return y
def At_DualCamera_torch_withshift(y, Phi, camera_respoce_curve, split_y_row, shift_func, shift_back_func):
    """
    :param y: [y1,y2] two observations concatenated vertically
    :param Phi: shifted Phi
    :param camera_respoce_curve: spectral response curve
    :param split_y_row: boundary between y1 and y2 rows
    :param shift_func: forward shift operator
    :return:
    """
    y = torch.unsqueeze(y, 1).repeat(1, Phi.shape[1], 1, 1)
    temp1,temp2 = y[:,:,:,:split_y_row],y[:,:,:,split_y_row:]

    x1 = temp1 * Phi
    x2 = temp2 * camera_respoce_curve
    x2 = shift_func(x2)
    # Dividing by the spectral response curve converges faster
    return x1+x2

####Unshift mode operators#####
def A_DualCamera_torch(x, Phi, camera_response_curve, shift_func, shift_back_func):
    """
    Dual-camera forward operator, both x and Phi have been shifted
    :param x:
    :param Phi:
    :param camera_response_curve: [1,c,1,1]
    :param shift_back_func: Reverse shift operator
    :return:
    """
    x1 = shift_func(x)  # Compute using the CUDA version of shift_func
    y1 = torch.sum(x1 * Phi, dim=1)  # cassi_measurement
    x_response = x * camera_response_curve
    y2 = torch.sum(x_response, dim=1)  # pan_measurement
    if y1.shape[2] == y2.shape[2]:
        # Vertical shift
        y = torch.cat([y1, y2], dim=1)  # concat on vertical
    else:
        y = torch.cat([y1, y2], dim=2)  # concat on vertical
    return y
def At_DualCamera_torch(y,Phi,camera_respoce_curve,split_y_row,shift_func,shift_back_func):
    """
    :param y: [y1,y2] two observations concatenated vertically
    :param Phi: shifted Phi
    :param camera_respoce_curve: spectral response curve
    :param split_y_row: boundary between y1 and y2 rows
    :param shift_func: forward shift operator
    :return:
    """
    y = torch.unsqueeze(y, 1).repeat(1, Phi.shape[1], 1, 1)

    temp1,temp2 = y[:,:,:,:split_y_row],y[:,:,:,split_y_row:]

    x1 = temp1 * Phi
    x1 = shift_back_func(x1)
    x2 = temp2 * camera_respoce_curve
    # Dividing by the spectral response curve converges faster
    return x1+x2

class ShiftDirecttion:
    UP_DOWN = 0   # In step, 1 means down, -1 means up
    LEFT_RIGHT = 1 # In step, 1 means right, -1 means left

def get_shift(default_direction:ShiftDirecttion=ShiftDirecttion.LEFT_RIGHT):

    if default_direction == ShiftDirecttion.UP_DOWN:
        def shift(inputs, step):
            """
            Upward shift
            :param inputs:
            :param step:
            :return:
            """
            row, col, L = inputs.shape
            new_row = row + abs(step) * (L - 1)
            output = np.zeros((new_row, col , L))
            for i in range(L):
                if step < 0:
                    output[new_row-i*abs(step)-row:new_row-i*abs(step),:, i] = inputs[:, :, i]
                else:
                    output[new_row - i * abs(step) - row:new_row - i * abs(step), :, L-i-1] = inputs[:, :, L-i-1]
            return output


        def shift_back(inputs, step):
            row, col, L = inputs.shape
            new_row = row - abs(step) * (L - 1)
            output = np.zeros((new_row, col, L))
            for i in range(L):
                if step < 0:
                    output[:, :, i] = inputs[row-i*abs(step)-new_row:row-i*abs(step),:, i]
                else:
                    output[:, :, L-i-1] = inputs[row - i * abs(step) - new_row:row - i * abs(step), :, L-i-1]
            return output
        def shift_torch(inputs, step=1,index_tensor=None,output=None):
            """
            :param inputs: b,c,h,w x_cube
            :param step: 1 for down, -1 for up
            :param index_tensor: Use index to shift
            :param output: Output handle
            :return: Output handle
            """
            [bs, nC, row, col] = inputs.shape
            new_row = row + abs(step) * (nC - 1)
            if output is None:
                output = torch.zeros(bs, nC, new_row,col).cuda()
            else:
                output = output * 0
            if index_tensor is None:
                for i in [j for j in range(nC)]:
                    if step < 0:
                        output[:, i, new_row-i*abs(step)-row:new_row-i*abs(step), :] = inputs[:, i, :, :]
                    else:
                        output[:, nC-i-1, new_row-i*abs(step)-row:new_row-i*abs(step), :] = inputs[:, nC-i-1, :, :]

            else:
                output[index_tensor] = inputs.reshape(-1)
            return output

        def shift_back_torch(inputs, step=1,index_tensor=None,output=None):
            """
            :param inputs: b,c,h+step(L-1), x_shift_cube
            :param step: Shift: 1 for down, -1 for up
            :param index_tensor:
            :param output: Output handle
            :return:
            """
            [bs, nC, row, col] = inputs.shape
            new_row = row - abs(step) * (nC - 1)
            if output is None:
                output = torch.zeros(bs, nC, new_row, col).cuda()
            else:
                output = output * 0
            if index_tensor is None:
                for i in range(nC):
                    if step < 0:
                        output[:, i, :, :] = inputs[:, i, row-i*abs(step)-new_row:row-i*abs(step), :]
                    else:
                        output[:, nC-i-1, :, :] = inputs[:, nC-i-1, row - i * abs(step) - new_row:row - i * abs(step), :]
            else:
                return inputs[index_tensor].reshape(output.shape)
                output[:,:,:,:] = inputs[index_tensor].reshape(output.shape)
            return output
    else:
        #LEFT RIGHT
        def shift(inputs, step):
            """
            Shift function for left and right
            :param inputs: Input 3D array
            :param step: 1 means shift right by one step, -1 means shift left by one step
            :return: Result array after left/right shifting
            """
            row, col, L = inputs.shape
            new_col = col + abs(step) * (L - 1)
            output = np.zeros((row, new_col, L))

            for i in range(L):
                if step < 0:
                    output[:, new_col - i * abs(step) - col:new_col - i * abs(step), i] = inputs[:, :, i]
                else:
                    output[:, new_col - i * abs(step) - col:new_col - i * abs(step), L - i - 1] = inputs[:, :, L - i - 1]

            return output
        def shift_back(inputs, step):
            """
            Inverse operation function for left/right shifting
            :param inputs: Input 3D array
            :param step: 1 means shift right by one step, -1 means shift left by one step
            :return: Result array after inverse left/right shifting
            """
            row, col, L = inputs.shape
            new_col = col - abs(step) * (L - 1)
            output = np.zeros((row, new_col, L))
            for i in range(L):
                if step < 0:
                    output[:, :, i] = inputs[:, col - i * abs(step) - new_col:col - i * abs(step), i]
                else:
                    output[:, :, L - i - 1] = inputs[:, col - i * abs(step) - new_col:col - i * abs(step), L - i - 1]
            return output
        def shift_torch(inputs, step=1,index_tensor=None,output=None):
            """
            :param inputs: b,c,h,w x_cube
            :param step: 1 for right, -1 for left
            :param index_tensor: Use index to shift
            :param output: Output handle
            :return: Output handle
            """
            outshape = (inputs.shape[0], inputs.shape[1], inputs.shape[2], inputs.shape[3] + abs(step) * (inputs.shape[1] - 1))
            assert output is None or output.shape == outshape, f"shift ops error:inputs={inputs.shape},ouputs={output.shape}"
            [bs, nC, row, col] = inputs.shape
            new_col= col + abs(step) * (nC - 1)
            if output is None:
                output = torch.zeros(bs, nC, row,new_col).cuda()
            else:
                output = output * 0
            if index_tensor is None:
                for i in [j for j in range(nC)]:
                    if step < 0:
                        output[:, i, : ,new_col-i*abs(step)-col:new_col-i*abs(step)] = inputs[:, i, :, :]
                    else:
                        output[:, nC-i-1,:, new_col-i*abs(step)-col:new_col-i*abs(step)] = inputs[:, nC-i-1, :, :]
            else:
                inputs = inputs.reshape(-1)
                assert index_tensor[0].shape[0] == inputs.shape[0],f"shift ops error:{index_tensor[0].shape[0]}!={inputs.shape[0]}"
                output[index_tensor] = inputs
            return output


        def shift_back_torch(inputs, step=1, index_tensor=None, output=None):
            """
            :param inputs: b,c,h+step(L-1), x_shift_cube
            :param step: 1 for right, -1 for left
            :param index_tensor:
            :param output: Output handle
            :return:
            """
            outshape = (inputs.shape[0], inputs.shape[1], inputs.shape[2], inputs.shape[3] - abs(step) * (inputs.shape[1] - 1))
            assert output is None or output.shape == outshape,f"shift back ops error:inputs={inputs.shape},ouputs={output.shape}"
            [bs, nC, row, col] = inputs.shape
            new_col = col - abs(step) * (nC - 1)
            if output is None:
                output = torch.zeros(bs, nC, row, new_col).cuda()
            else:
                output = output * 0
            if index_tensor is None:
                for i in range(nC):
                    if step < 0:
                        output[:, i, :, :] = inputs[:, i, : ,col-i*abs(step)-new_col:col-i*abs(step)]
                    else:
                        output[:, nC - i - 1, :, :] = inputs[:, nC-i-1,:, col-i*abs(step)-new_col:col-i*abs(step)]
            else:
                assert index_tensor[0].shape[0] == output.shape[0]*output.shape[1]*output.shape[2]*output.shape[3],f"shift back ops error:{index_tensor[0].shape[0]}!={output.shape[0]*output.shape[1]*output.shape[2]*output.shape[3]}"
                return inputs[index_tensor].reshape(output.shape)
            return output
    return shift,shift_back,shift_torch,shift_back_torch

def numpy2torch(x:np.ndarray):
    if len(x.shape) == 2:
        x = torch.tensor(x)
        return torch.tensor(x).unsqueeze(0).unsqueeze(0)
    if len(x.shape) == 3:
        x = torch.tensor(x)
        x = np.transpose(x, (2,0,1))
        return torch.tensor(x).unsqueeze(0)
    if len(x.shape) == 4:
        x = np.transpose(x, (0, 3, 1, 2))
        x = torch.tensor(x)
        return x
    else:
        return torch.tensor(x)
def singleton(cls):
    instances = {}
    def getinstance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return getinstance
