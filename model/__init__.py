def model_generator(opt,physicalModel):
    if opt.model_name == "CASSI_BENCHMARK_CST":
        print("CASSI_BENCHMARK_CST")
        from .cassi.benchmark.CST_adaptor import Adaptor as Model
        model = Model(opt.model_param,physicalModel=physicalModel).cuda()
    elif opt.model_name == "CASSI_BENCHMARK_MST":
        print("CASSI_BENCHMARK_MST")
        from model.cassi.benchmark.MST_adaptor import Adaptor as Model
        model = Model(opt.model_param,physicalModel=physicalModel).cuda()
    elif opt.model_name == "CASSI_BENCHMARK_MST.2":
        print("CASSI_BENCHMARK_MST")
        from model.cassi.benchmark.MST_adaptor_2 import Adaptor as Model
        model = Model(opt.model_param,physicalModel=physicalModel).cuda()
    elif opt.model_name == "CASSI_BENCHMARK_MSTPP":
        print("CASSI_BENCHMARK_MSTPP")
        from model.cassi.benchmark.mstpp_adaptor import Adaptor as Model
        model = Model(opt.model_param, physicalModel=physicalModel).cuda()
    elif opt.model_name == "CASSI_BENCHMARK_HDNET":
        print("CASSI_BENCHMARK_HDNET")
        from model.cassi.benchmark.hdnet_adaptor import Adaptor as Model
        model = Model(opt.model_param,physicalModel=physicalModel).cuda()
    elif opt.model_name == "CASSI_BENCHMARK_BISRNET":
        print("CASSI_BENCHMARK_BISRNET")
        from model.cassi.benchmark.bisrnet_adaptor import Adaptor as Model
        model = Model(opt.model_param,physicalModel=physicalModel).cuda()
    elif opt.model_name == "CASSI_BENCHMARK_DAUHST":
        print("CASSI_BENCHMARK_DAUHST")
        from model.cassi.benchmark.dauhst_adaptor import Adaptor as Model
        model = Model(opt.model_param,physicalModel=physicalModel).cuda()
    elif opt.model_name == "CASSI_BENCHMARK_PADUT":
        print("CASSI_BENCHMARK_PADUT")
        from model.cassi.benchmark.padut_adaptor import Adaptor as Model
        model = Model(opt.model_param,physicalModel=physicalModel).cuda()
    elif opt.model_name == "CASSI_BENCHMARK_RDLUF":
        print("CASSI_BENCHMARK_RDLUF")
        from model.cassi.benchmark.rdluf_adaptor import Adaptor as Model
        model = Model(opt.model_param,physicalModel=physicalModel).cuda()

    elif opt.model_name == "CASSI_BENCHMARK_SSR":
        print("CASSI_BENCHMARK_SSR")
        from model.cassi.benchmark.ssr_adaptor import Adaptor as Model
        model = Model(opt.model_param, physicalModel=physicalModel).cuda()





    elif opt.model_name == "CASSI_BENCHMARK_DSSP":
        print("CASSI_BENCHMARK_DSSP")
        from model.cassi.benchmark.dssp_adaptor import Adaptor as Model
        model = Model(opt.model_param, physicalModel=physicalModel).cuda()


    elif opt.model_name == "CASSI_BENCHMARK_DPU":
        print("CASSI_BENCHMARK_DPU")
        from model.cassi.benchmark.dpu_adaptor import Adaptor as Model
        model = Model(opt.model_param, physicalModel=physicalModel).cuda()






 

    else:
        raise Exception("model_not_find")
    return model