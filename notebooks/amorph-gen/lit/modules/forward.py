import torch
import numpy as np
import lightning.pytorch as pl

####################### Model #######################

from torch import nn, Tensor
from graphite.nn import MLP

class XANESDecoder(nn.Module):
    def __init__(self, node_dim, out_dim):
        super().__init__()
        self.node_dim = node_dim
        self.out_dim = out_dim
        self.decoder = MLP([node_dim, node_dim, out_dim], act=nn.SiLU())
    
    def forward(self, h_node:Tensor) -> Tensor:
        out = self.decoder(h_node)
        return torch.nn.functional.softplus(out)

class XANESMeshGraphNets(nn.Module):
    def __init__(self, encoder, processor, decoder):
        super().__init__()
        self.encoder   = encoder
        self.processor = processor
        self.decoder   = decoder
    
    def forward(self, x:Tensor, edge_index:Tensor, edge_attr:Tensor) -> Tensor:
        h_node, h_edge = self.encoder(x, edge_attr)
        h_node, h_edge = self.processor(h_node, edge_index, h_edge)
        return self.decoder(h_node)


####################### LightningModule #######################

from graphite.nn.models.mgn import Encoder, Processor, MeshGraphNets

class LitXANESNet(pl.LightningModule):
    def __init__(self, num_species, num_convs, dim, out_dim, ema_decay=0.999, learn_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()
        
        # Core model
        self.model = XANESMeshGraphNets(
            encoder   = Encoder(num_species, 3+1, dim, dim),
            processor = Processor(num_convs, dim, dim),
            decoder   = XANESDecoder(dim, out_dim),
        )

        # EMA model
        ema_avg = lambda avg_params, params, num_avg: ema_decay*avg_params + (1-ema_decay)*params
        self.ema_model = torch.optim.swa_utils.AveragedModel(self.model, avg_fn=ema_avg)

        # Training parameters
        self.learn_rate = learn_rate

    def training_step(self, batch, batch_idx):
        pred_y = self.model(batch.z, batch.edge_index, batch.edge_attr)
        train_loss = torch.nn.functional.mse_loss(pred_y[batch.train_mask], batch.y[batch.train_mask])
        self.log('train_loss', train_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch.num_graphs)
        valid_loss = torch.nn.functional.mse_loss(pred_y[~batch.train_mask], batch.y[~batch.train_mask])
        self.log('valid_loss', valid_loss, on_step=False, on_epoch=True, prog_bar=True,  batch_size=batch.num_graphs)
        self.log('hp_metric',  valid_loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch.num_graphs)
        return train_loss

    # def validation_step(self, batch, batch_idx):
    #     pred_y = self.model(batch.z, batch.edge_index, batch.edge_attr)
    #     loss = torch.nn.functional.mse_loss(pred_y[~batch.train_mask], batch.y[~batch.train_mask])
    #     self.log('valid_loss', loss, on_step=False, on_epoch=True, prog_bar=True,  batch_size=batch.num_graphs)
    #     self.log('hp_metric',  loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch.num_graphs)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learn_rate)

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.ema_model.update_parameters(self.model)