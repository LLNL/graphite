import torch
import numpy as np

# Typing
from torch import Tensor
from typing import Tuple, List, Optional

####################### Dataset #######################

from torch_geometric.data import Data, Dataset
from sklearn.preprocessing import OneHotEncoder
from scipy.spatial.transform import Rotation
import h5py

class MolTrajDataset(Dataset):
    def __init__(self, h5_fnames: List[str], interval: int = 1, scale: float = 1.0) -> None:
        super().__init__()
        self.h5_fnames = h5_fnames
        self.interval  = interval
        self.scale     = scale

        # Figure out global-to-local indexing for map-style dataset implementation
        self.global_idx = []
        self.local_idx  = []
        self.pos_trajs  = []
        for i, fname in enumerate(h5_fnames):
            file     = h5py.File(fname, 'r')
            pos_traj = file['pos']
            traj_len = len(pos_traj) - interval
            self.pos_trajs  += [pos_traj]
            self.global_idx += [i]*traj_len
            self.local_idx  += list(range(traj_len))

        # Prepare node/atom encoder
        self.all_symbols = np.concatenate([pos.attrs['symbols'] for pos in self.pos_trajs])
        self.node_encoder = OneHotEncoder(sparse_output=False)
        self.node_encoder.fit(self.all_symbols.reshape(-1, 1))

    def len(self):
        return len(self.global_idx)
    
    def get(self, idx):
        global_idx = self.global_idx[idx]
        local_idx  = self.local_idx[idx]
        pos_traj   = self.pos_trajs[global_idx]

        z  = pos_traj.attrs['symbols'].reshape(-1, 1)
        z  = self.node_encoder.transform(z)

        pos   = pos_traj[local_idx]
        disp  = pos_traj[local_idx + self.interval] - pos
        pos  *= self.scale
        disp *= self.scale
        
        data = Data(
            pos  = torch.tensor(pos,  dtype=torch.float),
            disp = torch.tensor(disp, dtype=torch.float),
            z    = torch.tensor(z,    dtype=torch.float),
        )
        data = self._random_rotate(data)
        return data

    def _random_rotate(self, data):
        R = torch.tensor(Rotation.random().as_matrix(), dtype=torch.float, device=data.pos.device)
        data.pos  @= R
        data.disp @= R
        return data

####################### LightningDataModule #######################

import lightning as L
from torch_geometric.loader import DataLoader

class MolTrajDataModule(L.LightningDataModule):
    def __init__(self, file_list: List[str], interval: int = 1, scale: float = 1.0, batch_size: int = 8, num_workers: int = 4) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.file_list    = file_list
        self.interval     = interval
        self.scale        = scale
        self.batch_size   = batch_size
        self.num_workers  = num_workers

    def prepare_data(self):
        # Download, IO, etc. Useful with shared filesystems
        # Only called on 1 GPU/TPU in distributed
        pass

    def setup(self, stage=None):
        # Make assignments here (val/train/test split)
        # Called on every process in DDP
        self.dataset = MolTrajDataset(self.file_list, interval=self.interval, scale=self.scale)
        self.train_set, self.valid_set = torch.utils.data.random_split(
            self.dataset, [0.9, 0.1], generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(self.train_set, shuffle=True,  batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.valid_set, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def teardown(self, stage=None):
        # Clean up state after the trainer stops, delete files...
        # Called on every process in DDP
        pass
