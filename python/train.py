# --------------------------------------------------------
# train.py
# by SangHyuk Yoo, shyoo@yonsei.ac.kr
# last modified : Sat Aug 20 12:13:23 KST 2022
#
# Objectives
# Train force predictor
# 
# Prerequisites library
# 1. ASE(Atomistic Simulation Environment)
# 2. DGL
# 3. PyTorch
# 4. scikit-learn
# 5. PyTorch-ligthning
#
# Usage 
# python3 train.py --input-file train_info.json
# --------------------------------------------------------

import argparse
import json
import os
from typing import Optional

import numpy as np

import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from dgl.dataloading import GraphDataLoader
from sklearn.preprocessing import StandardScaler

from graph import MDDataset
from graph_net import MDNet

class LightDataModule(pl.LightningDataModule):
    def __init__(self, train_params, data_params):
        super().__init__()
        self.raw_dir = data_params['raw_dir']
        self.save_dir = data_params['save_dir']
        self.traj_index = data_params['traj_index']
        self.snap_index = data_params['snap_index']
        self.r_cut = data_params['r_cut']
        self.force_reload = data_params['force_reload']
        self.traj_index_list = range(self.traj_index['start'],
                                     self.traj_index['end'],
                                     self.traj_index['step'])
        self.snap_index_list = range(self.snap_index['start'],
                                     self.snap_index['end'],          
                                     self.snap_index['step'])
        self.train_test_ratio = train_params['train_test_ratio']
        self.batch_size = train_params['batch_size']
        self.shuffle = train_params['shuffle']

    def setup(self, stage: Optional[str] = None):
        # create main dataset
        self.dataset = MDDataset(url=None,
                                 raw_dir=self.raw_dir,
                                 save_dir=self.save_dir,
                                 force_reload=self.force_reload,
                                 verbose=False,
                                 r_cut=self.r_cut,
                                 traj_index=self.traj_index,
                                 snap_index=self.snap_index)

        # preapre train and test set
        generator = torch.Generator()
        num_data = len(self.dataset)
        num_train = torch.ceil(torch.tensor(self.train_test_ratio*num_data)).long()
        num_test  = num_data - num_train
        self.train_set, self.test_set = random_split(self.dataset, 
                                                     [num_train, num_test],
                                                     generator)

    def train_dataloader(self):
        train_dataloader = GraphDataLoader(self.train_set,
                                           num_workers=8,
                                           batch_size=self.batch_size,
                                           shuffle=True)
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = GraphDataLoader(self.test_set,
                                         num_workers=8,
                                         batch_size=self.batch_size,
                                         shuffle=False)
        return val_dataloader


class LightModel(pl.LightningModule):
    def __init__(self, train_params, model_params):
        super().__init__()

        # get train parameters
        self.lr = train_params['learning_rate']
        self.min_epochs = train_params['min_epochs']
        self.max_epochs = train_params['max_epochs']

        # get model parameters
        self.model_params = model_params
        embed_feats = self.model_params['embed_feats']
        hidden_feats = self.model_params['hidden_feats']
        num_blocks = self.model_params['num_blocks']
        use_layer_norm = self.model_params['use_layer_norm']

        # create a model
        self.model = MDNet(in_node_feats=embed_feats, 
                           embed_feats=embed_feats,
                           out_node_feats=embed_feats,
                           hidden_feats=hidden_feats,
                           num_blocks=num_blocks,
                           use_layer_norm=use_layer_norm)

        # create scaler
        self.scaler = StandardScaler()
        self.avg_force = nn.Parameter(torch.tensor([0.]), requires_grad=False)
        self.std_force = nn.Parameter(torch.tensor([1.]), requires_grad=False)

        # set loss function
        self.mae_fn = nn.L1Loss(reduction='mean')
        self.mse_fn = nn.MSELoss(reduction='mean')

    def forward(self, batch):
        return self.model(batch[0])

    def training_step(self, batch, batch_idx):
        # predict forces
        batched_graph, _ = batch
        preds = self.model(batched_graph)
        
        # scale the references
        refs = batched_graph.ndata['forces'].detach().cpu()
        natoms, dims = refs.shape
        refs_flat = np.asarray(refs.reshape((-1, 1)))
        self.scaler.partial_fit(refs_flat)
        refs_flat = self.scaler.transform(refs_flat)
        refs = torch.from_numpy(refs_flat).view(natoms, dims)
        if not refs.is_cuda:
            refs = refs.cuda()

        # save scaler value
        self.avg_force[0] = self.scaler.mean_[0]
        self.std_force[0] = self.scaler.scale_[0]

        # calculate MAE
        loss = self.mae_fn(preds, refs)
        loss = loss + 1e-3*torch.abs(torch.mean(preds))

        # log
        self.log('train mae', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('avg_force', self.avg_force[0], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('std_force', self.std_force[0], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        batched_graph, _ = batch
        with torch.no_grad():
            # predict forces
            preds = self.model(batched_graph)

            # scale the refrences
            refs = batched_graph.ndata['forces'].cpu().detach()
            natoms, dims = refs.shape
            refs_flat = np.asarray(refs.reshape((-1, 1)))
            self.scaler.partial_fit(refs_flat)
            refs_flat = self.scaler.transform(refs_flat)
            refs = torch.from_numpy(refs_flat).view(natoms, dims)
            if not refs.is_cuda:
                refs = refs.cuda()

            # calculate loss: MAE and RMSE
            mae = self.mae_fn(preds, refs)
            mse = self.mse_fn(preds, refs)
            
            # log
            self.log('val mae', mae, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log('val mse', mse, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        sched = StepLR(optim, step_size=5, gamma=0.001**(5/self.max_epochs))
        return [optim], [sched]


def main(inputs):
    train_params = inputs['train_params']
    model_params = inputs['model_params']
    data_params = inputs['data_params']

    # define reproductivity
    pl.seed_everything(42, workers=True)

    # define data module
    dm = LightDataModule(train_params=train_params,
                         data_params=data_params)
    dm.setup()

    # define model
    model = LightModel(train_params=train_params,
                       model_params=model_params)

    # prepare logger
    wandb_logger = pl.loggers.WandbLogger(project='mdnet',
                                          log_model=True)

    # define checkpoint model
    dirname_save = train_params['save_dir']
    if not os.path.exists(dirname_save):
        os.makedirs(dirname_save)
    filename_ckpt = 'best'
    ckpt_callback = ModelCheckpoint(save_top_k=1,
                                    monitor='val mae',
                                    mode="min",
                                    dirpath=dirname_save,
                                    filename=filename_ckpt,
                                    every_n_epochs=1,
                                    save_on_train_epoch_end=False)

    # train model
    min_epochs = train_params['min_epochs']
    max_epochs = train_params['max_epochs']
    trainer = pl.Trainer(accelerator='gpu',
                         devices=1, 
                         min_epochs=min_epochs, 
                         max_epochs=max_epochs,
                         logger=wandb_logger,
                         default_root_dir=dirname_save,
                         callbacks=[ckpt_callback],
                         deterministic=True)
    trainer.fit(model=model, 
                train_dataloaders=dm.train_dataloader(),
                val_dataloaders=dm.val_dataloader())
    trainer.save_checkpoint(filepath=os.path.join(dirname_save, 'last.ckpt'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', help='input file', type=str, default='train_info.json')
    args = parser.parse_args()

    with open(args.input_file, 'r') as fh:
        input_params = json.load(fh)

    main(input_params)