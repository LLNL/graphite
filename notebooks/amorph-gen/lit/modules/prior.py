import torch
import numpy as np
import lightning.pytorch as pl

####################### Model #######################

from torch import nn, Tensor

class ScoreModel(nn.Module):
    def __init__(self, encoder, processor, decoder):
        super().__init__()
        self.encoder   = encoder
        self.processor = processor
        self.decoder   = decoder
    
    def forward(self, x:Tensor, edge_index:Tensor, edge_attr:Tensor, t:Tensor, sigma:Tensor) -> Tensor:
        h_node, h_edge = self.encoder(x, edge_attr, t)
        h_node, h_edge = self.processor(h_node, edge_index, h_edge)
        return self.decoder(h_node) / sigma

####################### LightningModule #######################

from graphite.nn.models.mgn import Encoder_dpm, Processor, Decoder

class LitScoreNet(pl.LightningModule):
    def __init__(self, num_species, num_convs, dim, ema_decay, learn_rate):
        super().__init__()
        self.save_hyperparameters()
        
        # Core model (MGN)
        self.model = ScoreModel(
            encoder   = Encoder_dpm(num_species, 3+1, dim, dim),
            processor = Processor(num_convs, dim, dim),
            decoder   = Decoder(dim, 3),
        )

        # EMA model
        ema_avg = lambda avg_params, params, num_avg: ema_decay*avg_params + (1-ema_decay)*params
        self.ema_model = torch.optim.swa_utils.AveragedModel(self.model, avg_fn=ema_avg)

        # Training parameters
        self.learn_rate = learn_rate

    def training_step(self, batch, batch_idx):
        score = self.model(batch.z, batch.edge_index, batch.edge_attr, batch.t, batch.sigma_r)
        loss = (score*batch.sigma_r + batch.eps_r).pow(2).sum(dim=-1).mean()
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True,  batch_size=batch.num_graphs)
        self.log('hp_metric',  loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch.num_graphs)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learn_rate)
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.ema_model.update_parameters(self.model)