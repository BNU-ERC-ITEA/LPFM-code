#from .service.VideoDCCHIPhysicalGenerator import VideoDCCHIPhysicalGenerator
#from .service.DAUCASSIPhysical_PSF import DAUCASSIPhysical as DAUCASSIPhysicalPSF


class NonPhysicalGenerator():
    def __init__(self, opt):
        self.opt = opt
    def forward(self,batch):
        return batch


def physical_generator(opt):
    if opt.physical_name == "DAUCASSIPairDatasetPhysical":
        from .service.DAUCASSIPairDatasetPhysical import DAUCASSIPhysical as DAUCASSIPairDatasetPhysical
        return DAUCASSIPairDatasetPhysical(opt)
  
    else:
        raise Exception("physical_not_find")


    # if 'mst' in opt.model_name:
    #     from .MST import MST as Model
    #     model = Model(in_dim=3,out_dim=3,stage=1).cuda()
    #     return model
    # else:
    #     raise Exception("model_not_find")