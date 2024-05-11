import torch
import numpy as np

####################### LightningModule #######################

import lightning as L
import torch.nn.functional as F
from graphite.nn.models.equivariant_transformer import Encoder, Processor, Decoder, EquivariantTransformer

class LitEquivariantNoiseNet(L.LightningModule):
    def __init__(self, num_species, node_dim, init_edge_dim, edge_dim, ff_dim, num_heads, num_layers, sigma_max, cutoff, learn_rate):
        super().__init__()
        self.save_hyperparameters()

        # Core model
        self.model = EquivariantTransformer(
            encoder   = Encoder(num_species=num_species, node_dim=node_dim, init_edge_dim=init_edge_dim, edge_dim=edge_dim),
            processor = Processor(num_convs=num_layers, node_dim=node_dim, num_heads=num_heads, ff_dim=ff_dim, edge_dim=edge_dim),
            decoder   = Decoder(dim=node_dim, num_scalar_out=1, num_vector_out=1),
        )

        # For later use
        self.sigma_max  = sigma_max
        self.cutoff     = cutoff
        self.learn_rate = learn_rate

    def _rattle_particles(self, batch, sigma_max, sigma_min=0.001):
        # Determine a different noise level per batch
        sigma = torch.empty(batch.num_graphs, device=batch.species.device).uniform_(sigma_min, sigma_max)
        sigma = sigma[batch.batch, None]
        
        # Add noise
        batch.disp = sigma * torch.randn_like(batch.pos)
        batch.pos += batch.disp
        i, j = batch.edge_index
        batch.edge_vec += batch.disp[j] - batch.disp[i]
        edge_len = torch.linalg.norm(batch.edge_vec, dim=1, keepdim=True)
        batch.edge_attr = edge_len
        return batch

    def _downselect_edges(self, batch, cutoff):
        edge_index, edge_attr, edge_vec = batch.edge_index, batch.edge_attr, batch.edge_vec
        edge_len = torch.linalg.norm(edge_vec, dim=1)
        mask = (edge_len < cutoff)
        batch.edge_index = edge_index[:, mask]
        batch.edge_attr  = edge_attr[mask]
        batch.edge_vec   = edge_vec[mask]
        return batch

    def _get_loss(self, batch):
        batch = self._rattle_particles(batch, sigma_max=self.sigma_max)
        batch = self._downselect_edges(batch, cutoff=self.cutoff)
        _, pred_disp = self.model(batch.species, batch.edge_index, batch.edge_attr, batch.edge_vec)
        return F.mse_loss(pred_disp, batch.disp)

    def training_step(self, batch, batch_idx):
        loss = self._get_loss(batch)
        self.log('train_loss', loss, batch_size=batch.num_graphs, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self._get_loss(batch)
        self.log('valid_loss', loss, batch_size=batch.num_graphs, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learn_rate)