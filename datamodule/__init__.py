from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader



class DataModule(LightningDataModule):
    def __init__(self,batch_size=1,trainSet=None,testSet=None,collate_fn=None):
        self.trainSet = trainSet
        self.testSet = testSet
        self.collate_fn = collate_fn
        self.batch_size = batch_size
        super().__init__()

    def setup(self, stage):
        collate_fn = self.collate_fn
        trainLoader = DataLoader(self.trainSet, batch_size=self.batch_size,
                                 shuffle=False,collate_fn=collate_fn)
        testLoader = DataLoader(self.testSet, batch_size=self.batch_size,
                                shuffle=False,collate_fn=collate_fn)
        # testLoader = DataLoader(self.testSet, batch_size=1,
        #                         shuffle=False)
        self.train = trainLoader
        self.val = testLoader

    def train_dataloader(self):
        return self.train

    def val_dataloader(self):
        return self.val

    def test_dataloader(self):
        return self.val


def dataset_generator(opt,collate_fn=None,physical_model=None):
    train_set, test_set = None,None
    if 'div2k_mxt_dataset' in opt.dataset_name:
        from .div2k_mxt_dataset import DIV2KDataSet as Dataset
        train_set,test_set = Dataset(dataset_folder=opt.train_root,max_num=opt.train_max_num),Dataset(dataset_folder=opt.valid_root,max_num=opt.test_max_num)
        # return train_set,test_set3

    elif opt.dataset_name == "pair_csv_cassi_wphi":
        from datamodule.pair_datasets import load_pair_csv_cassi_wphi
        train_set, test_set = load_pair_csv_cassi_wphi(physical_model, opt)

    else:
        raise Exception("dataset_not_find")
    assert train_set is not None and test_set is not  None
    dm = DataModule(opt.batch_size,train_set,test_set,collate_fn=collate_fn)
    return dm
