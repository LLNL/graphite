import torch
import numpy as np

####################### Dataset #######################

from torch_geometric.data import Data, Dataset
from sklearn.preprocessing import LabelEncoder
from graphite.nn import periodic_radius_graph

class PeriodicStructureDataset(Dataset):
    def __init__(self, atoms_list, large_cutoff, duplicate=128):
        super().__init__()
        
        self.dataset = []
        for atoms in atoms_list:
            species = LabelEncoder().fit_transform(atoms.numbers)  # this line implies that only single-element systems are considered

            species = torch.tensor(species,             dtype=torch.long)
            pos     = torch.tensor(atoms.positions,     dtype=torch.float)
            cell    = torch.tensor(atoms.cell.tolist(), dtype=torch.float)
            
            edge_index, edge_vec = periodic_radius_graph(pos, large_cutoff, cell)
            data = Data(species=species, edge_index=edge_index, edge_vec=edge_vec, pos=pos)
            self.dataset.append(data)
        
        self.dataset = [d.clone() for d in self.dataset for _ in range(duplicate)]
    
    def len(self):
        return len(self.dataset)
    
    def get(self, idx):
        return self.dataset[idx].clone()

####################### LightningDataModule #######################

import lightning as L
import ase.io
from torch_geometric.loader import DataLoader

class PeriodicStructureDataModule(L.LightningDataModule):
    def __init__(self, file_list, large_cutoff, duplicate=128, batch_size=8, num_workers=4):
        super().__init__()
        self.save_hyperparameters()

        self.file_list    = file_list
        self.large_cutoff = large_cutoff
        self.duplicate    = duplicate
        self.batch_size   = batch_size
        self.num_workers  = num_workers

    def prepare_data(self):
        # Download, IO, etc. Useful with shared filesystems
        # Only called on 1 GPU/TPU in distributed
        pass

    def setup(self, stage=None):
        # Make assignments here (val/train/test split)
        # Called on every process in DDP
        atoms_list = [ase.io.read(f) for f in self.file_list]
        self.dataset = PeriodicStructureDataset(atoms_list, large_cutoff=self.large_cutoff, duplicate=self.duplicate)
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
