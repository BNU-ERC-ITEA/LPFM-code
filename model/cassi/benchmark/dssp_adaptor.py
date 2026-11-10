from model.cassi.benchmark.base_net.dssp import Net
from model.cassi.benchmark.adaptor_tool import shift_factory, pad_cassi_data, prepare_input_data, PAD_MODE, get_pad
from physical.service.DAUCASSIPairDatasetPhysical import DAUCASSIPhysical
import torch
import torch.nn.functional as F
mse = torch.nn.MSELoss().cuda()

class Adaptor(Net):
    def __init__(self, opt,physicalModel:DAUCASSIPhysical=None):
        self.physicalModel = physicalModel
        self.input_keys = ["Y","MASK","Phi","Phi_s"]
        self.pad_keys = ["Y","MASK","Phi","Phi_s"]
        self.train_mode = True
        self.opt = opt

        # Initialize parameters and operators
        self._init_configurations()
        self._prepare_shift_operators()

        # Call parent class initialization
        super().__init__(**opt)

        self._init_custom_nn()
        self._prepare_temp_variables()
    def _init_custom_nn(self):
        pass

    def _init_configurations(self):
        """Initialize all configuration parameters"""
        self.shift_config = {}
        self._update_mode_specific_params()
        self._prepare_padding_metadata()

    def _prepare_temp_variables(self):
        self.result = None

    def _update_mode_specific_params(self):
        """Update parameters based on current mode"""
        if self.train_mode:
            cube_size = self.physicalModel.cassi_model.cube_size
            phi_shape = self.physicalModel.cassi_model.phi.shape
            self.key_meta = ""
        else:
            cube_size = self.physicalModel.cassi_model.test_cube_size
            phi_shape = self.physicalModel.cassi_model.test_phi.shape
            self.key_meta = "test"

        self.cube_shape = cube_size
        self.cube_shift_shape = phi_shape

        self.shift_config.update({
            "cube_size": cube_size,
            "shift_size": self.physicalModel.cassi_model.shift_size,
            "shift_step": self.physicalModel.cassi_model.shift_step
        })

    def _prepare_padding_metadata(self):
        """Prepare padding-related metadata"""
        self.pad_cube_shape, self.pad_cube_shift_shape, self.pad_h, self.pad_w = get_pad(
            self.cube_shape,
            self.cube_shift_shape,
            PAD_MODE.H,
            32
        )

        self.pad_shift_config = {
            "cube_size": self.pad_cube_shape,
            "shift_size": self.shift_config["shift_size"],
            "shift_step": self.shift_config["shift_step"]
        }

        if self.train_mode:
            self._log_padding_info()

    def _log_padding_info(self):
        """Log padding information"""
        print(f"""
        Padding Information:
        - Cube shape:    {self.cube_shape} => {self.pad_cube_shape}
        - Shift shape:   {self.cube_shift_shape} => {self.pad_cube_shift_shape}
        - Pad dimensions: height={self.pad_h}, width={self.pad_w}
        """)

    def _prepare_shift_operators(self):
        """Prepare shift operators"""
        self.shift1, self.shift_back1 = shift_factory.get_two_operator(
            self.pad_shift_config,
            self.key_meta+"net_operator"
        )

    def shift_back(self, x, len_shift=2):
        """Reverse shift operation"""
        return self.shift_back1(x)

    def shift(self, x, len_shift=2):
        """Forward shift operation"""
        return self.shift1(x)

    def set_train_mode(self):
        """Set to training mode"""
        self.train_mode = True
        self._init_configurations()
        self._prepare_shift_operators()

    def set_test_mode(self):
        """Set to test mode"""
        self.train_mode = False
        self._init_configurations()
        self._prepare_shift_operators()

    def forward(self, y, input_mask=None):
        """
        Forward pass

        Args:
            y: [b,256,310] input data
            input_mask: tuple containing Phi and Phi_s

        Returns:
            z_crop: [b,28,256,256] cropped result
        """
        Phi, Phi_s = input_mask

        # Prepare input data
        data = prepare_input_data(
            y, Phi,
            self.shift_config,
            self.input_keys,
            key_meta=self.key_meta
        )

        # Apply padding
        if self.pad_h != 0 or self.pad_w != 0:
            data = pad_cassi_data(
                data,
                self.pad_h,
                self.pad_w,
                self.pad_keys
            )

        # Call parent forward pass and crop result
        result = self.base_forward(data)
        result = result[:, :, :self.cube_shape[2], :self.cube_shape[3]]
        self.result = result
        return result

    def base_forward(self, data):
        return super().forward(data['Y'].unsqueeze(1),(data["Phi"], data["Phi_s"].unsqueeze(1)))

