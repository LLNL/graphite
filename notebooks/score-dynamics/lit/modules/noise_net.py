import torch
import numpy as np

# Typing
from torch import Tensor
from typing import Tuple, List, Optional

####################### Base modules #######################

from torch import nn
from graphite.nn.basis import GaussianRandomFourierFeatures
from graphite.nn import MLP

class ScoreDynamicsModel(nn.Module):
    def __init__(self, encoder, processor, decoder):
        super().__init__()
        self.encoder   = encoder
        self.processor = processor
        self.decoder   = decoder
    
    def forward(self, x_atm:Tensor, bnd_index:Tensor, x_bnd:Tensor, x:Tensor, t:Tensor) -> Tensor:
        h_atm, h_bnd = self.encoder(x_atm, x_bnd, x, t, bnd_index)
        h_atm, h_bnd = self.processor(h_atm, bnd_index, h_bnd)
        return self.decoder(h_atm)

class ScoreDynamicsEncoder(nn.Module):
    def __init__(self, init_atm_dim, init_bnd_dim, dim=128):
        super().__init__()        
        self.init_atm_dim = init_atm_dim
        self.init_bnd_dim = init_bnd_dim
        self.dim          = dim
        
        self.embed_atm = nn.Sequential(MLP([init_atm_dim, dim, dim], act=nn.SiLU()), nn.LayerNorm(dim))
        self.embed_bnd = nn.Sequential(MLP([init_bnd_dim, dim, dim], act=nn.SiLU()), nn.LayerNorm(dim))
        self.embed_time = nn.Sequential(
            GaussianRandomFourierFeatures(dim, input_dim=1),
            MLP([dim, dim, dim], act=nn.SiLU()), nn.LayerNorm(dim),
        )
    
    def forward(self, x_atm:Tensor, x_bnd:Tensor, dx:Tensor, t:Tensor, bnd_index:Tensor) -> Tuple[Tensor, Tensor]:
        # Embed atoms (and add time embedding)
        h_atm = self.embed_atm(x_atm)
        h_atm += self.embed_time(t)

        # Embed bonds
        i = bnd_index[0]
        j = bnd_index[1]
        disp_bnd_vec = x_bnd[:, :3] + dx[j] - dx[i]
        disp_bnd_len = torch.linalg.norm(disp_bnd_vec, dim=-1, keepdim=True)
        h_bnd = self.embed_bnd(torch.hstack([x_bnd, disp_bnd_vec, disp_bnd_len]))
        
        return h_atm, h_bnd

####################### LightningModule #######################

import lightning as L
import torch.nn.functional as F
from torch_geometric.nn import radius_graph

from graphite.nn.models.mgn import Processor, Decoder 
from graphite.diffusion import DPMSolverDiffuser

class LitNoiseNet(L.LightningModule):
    def __init__(self, num_species: int, num_convs: int, dim: int, out_dim: int, cutoff: float = 4.0, ema_decay: float = 0.9999, learn_rate: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()

        # Core model
        self.model = ScoreDynamicsModel(
            encoder   = ScoreDynamicsEncoder(num_species, 4*2, dim),
            processor = Processor(num_convs, dim, dim),
            decoder   = Decoder(dim, out_dim),
        )

        # EMA model
        ema_avg = lambda avg_params, params, num_avg: ema_decay*avg_params + (1-ema_decay)*params
        self.ema_model = torch.optim.swa_utils.AveragedModel(self.model, avg_fn=ema_avg)

        # Diffuser
        self.diffuser = DPMSolverDiffuser(schedule='linear')

        # Parameters
        self.learn_rate = learn_rate
        self.cutoff = cutoff

    def _molecular_graph(self, batch, cutoff):
        batch.edge_index = radius_graph(batch.pos, r=cutoff, batch=batch.batch)
        i, j = batch.edge_index
        bond_vec = batch.pos[j] - batch.pos[i]
        bond_len = bond_vec.norm(dim=-1, keepdim=True)
        batch.edge_attr = torch.hstack([bond_vec, bond_len])
        return batch

    def _get_loss(self, batch):
        lambdas = torch.empty(batch.num_graphs, 1, device=batch.pos.device).uniform_(self.diffuser.lambda_max, self.diffuser.lambda_min)
        t       = self.diffuser.t_lambda(lambdas)[batch.batch]
        disp, eps = self.diffuser.forward_noise(batch.disp, t)
        pred_eps = self.model(batch.z, batch.edge_index, batch.edge_attr, disp, t)
        return nn.functional.mse_loss(pred_eps, eps)

    def training_step(self, batch, batch_idx):
        batch = self._molecular_graph(batch, cutoff=self.cutoff)
        loss = self._get_loss(batch)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch.num_graphs)
        return loss

    def validation_step(self, batch, batch_idx):
        batch = self._molecular_graph(batch, cutoff=self.cutoff)
        loss = self._get_loss(batch)
        self.log('valid_loss', loss, on_step=False, on_epoch=True, prog_bar=True,  batch_size=batch.num_graphs)
        self.log('hp_metric',  loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch.num_graphs)
    
    def configure_optimizers(self):
        return torch.optim.RAdam(self.model.parameters(), lr=self.learn_rate)

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.ema_model.update_parameters(self.model)
