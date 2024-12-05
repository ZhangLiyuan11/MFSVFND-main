from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model.MFSVFND import MFSVFNDModel
from utils.dataloader import MFSVFNDDataset
from utils.tools import *
from utils.Trainer import Trainer
import numpy as np

def _init_fn(worker_id):
    np.random.seed(2024)

class Run():
    def __init__(self,config):
        self.dataset = config['dataset']
        self.epoches = config['epoches']
        self.batch_size = config['batch_size']
        self.num_workers = config['num_workers']
        self.epoch_stop = config['epoch_stop']
        self.seed = config['seed']
        self.device = config['device']
        self.lr = config['lr']
        self.lambd=config['lambd']
        self.save_param_dir = config['path_param']
        self.path_tensorboard = config['path_tensorboard']
        self.dropout = config['dropout']
        self.weight_decay = config['weight_decay']

    def get_dataloader_temporal(self):
        if self.dataset == 'fakesv':
            token = pretrain_bert_wwm_token()
            from utils.dataloader import fakesv_collate_fn as collate_fn
        else:
            token = pretrain_bert_uncased_token()
            from utils.dataloader import fakett_collate_fn as collate_fn

        dataset_train = MFSVFNDDataset('vid_time3_train.txt',token,self.dataset)
        dataset_val = MFSVFNDDataset('vid_time3_val.txt',token,self.dataset)
        dataset_test = MFSVFNDDataset('vid_time3_test.txt',token,self.dataset)

        train_dataloader = DataLoader(dataset_train, batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
            worker_init_fn=_init_fn,
            collate_fn=collate_fn)
        val_dataloader = DataLoader(dataset_val, batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            drop_last=True,
            worker_init_fn=_init_fn,
            collate_fn=collate_fn)
        test_dataloader=DataLoader(dataset_test, batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            drop_last=True,
            worker_init_fn=_init_fn,
            collate_fn=collate_fn)
 
        dataloaders =  dict(zip(['train', 'val', 'test'],[train_dataloader, val_dataloader, test_dataloader]))
        return dataloaders

    def get_model(self):
        self.model = MFSVFNDModel(fea_dim=128,dropout=self.dropout,dataset=self.dataset)
        return self.model

    def main(self):
        self.model = self.get_model()
        dataloaders = self.get_dataloader_temporal()
        trainer = Trainer(model=self.model, device = self.device, lr = self.lr, dataloaders = dataloaders, epoches = self.epoches, dropout = self.dropout, weight_decay = self.weight_decay,
                epoch_stop = self.epoch_stop, save_param_path = self.save_param_dir+self.dataset+"/", writer = SummaryWriter(self.path_tensorboard))
        result = trainer.train()
        return result

