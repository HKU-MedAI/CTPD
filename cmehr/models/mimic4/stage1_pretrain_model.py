from typing import Dict
import ipdb
import numpy as np
import sklearn.metrics as metrics
import torch
from torch import nn
import torch.nn.functional as F
from lightning import LightningModule
from cmehr.models.mimic4.UTDE_modules import multiTimeAttention
from cmehr.backbone.vision.pretrained import get_biovil_t_image_encoder
from lightning.pytorch.utilities.types import STEP_OUTPUT


class MIMIC4PretrainModule(LightningModule):
    def __init__(self,
                 orig_d_ts: int = 15,
                 orig_reg_d_ts: int = 30,
                 max_epochs: int = 10,
                 img_learning_rate: float = 1e-4,
                 ts_learning_rate: float = 4e-4,
                 period_length: int = 48,
                 embed_time: int = 64,
                 embed_dim: int = 128,
                 num_imgs: int = 5,
                 *args,
                 **kwargs
                 ):
        ''' Maybe to extract visual features offline ...'''
        # TODO: add more arguments for ablation study
        super().__init__()
        self.save_hyperparameters()
        
        self.orig_d_ts = orig_d_ts
        self.orig_reg_d_ts = orig_reg_d_ts
        self.max_epochs = max_epochs
        self.img_learning_rate = img_learning_rate
        self.ts_learning_rate = ts_learning_rate
        self.tt_max = period_length
        self.embed_dim = embed_dim
        self.num_imgs = num_imgs

        self.img_encoder = get_biovil_t_image_encoder()
        for param in self.img_encoder.parameters():
            param.requires_grad = False
        self.img_embed_dim = 512
        self.img_proj_layer = nn.Linear(self.img_embed_dim, embed_dim)

        # formulate the regular time stamps
        self.periodic = nn.Linear(1, embed_time-1)
        self.linear = nn.Linear(1, 1)
        self.time_query = torch.linspace(0, 1., self.tt_max)
        self.time_attn_ts = multiTimeAttention(
            self.orig_d_ts*2, self.embed_dim, embed_time, 8)
        self.time_attn_cxr = multiTimeAttention(
            self.embed_dim, self.embed_dim, embed_time, 8)
                
    def learn_time_embedding(self, tt):
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)
    
    def forward_ts_mtand(self,
                         x_ts: torch.Tensor,
                         x_ts_mask: torch.Tensor,
                         ts_tt_list: torch.Tensor):
        '''
        Forward irregular time series using mTAND.
        '''
        # (B, N) -> (B, N, embed_time)
        # fixed
        time_key_ts = self.learn_time_embedding(
            ts_tt_list)
        x_ts_irg = torch.cat((x_ts, x_ts_mask), 2)
        x_ts_mask = torch.cat((x_ts_mask, x_ts_mask), 2)

        time_query = self.learn_time_embedding(
            self.time_query.unsqueeze(0).type_as(x_ts))
        # query: (1, N_r, embed_time),
        # key: (B, N, embed_time),
        # value: (B, N, 2 * D_t)
        # mask: (B, N, 2 * D_t)
        # out: (B, N_r, 128?)
        proj_x_ts_irg = self.time_attn_ts(
            time_query, time_key_ts, x_ts_irg, x_ts_mask)

        return proj_x_ts_irg

    def forward_cxr_mtand(self,
                         x_cxr: torch.Tensor,
                         x_cxr_mask: torch.Tensor,
                         cxr_tt_list: torch.Tensor):
        '''
        Forward irregular CXRs using mTAND.
        '''
        # (B, N) -> (B, N, embed_time)
        # fixed
        time_key_cxr = self.learn_time_embedding(
            cxr_tt_list)
        time_query = self.learn_time_embedding(
            self.time_query.unsqueeze(0).type_as(x_cxr))
        # query: (1, N_r, embed_time),
        # key: (B, N, embed_time),
        # value: (B, N, 2 * D_t)
        # mask: (B, N, 2 * D_t)
        # out: (B, N_r, 128?)
        proj_x_cxr_irg = self.time_attn_cxr(
            time_query, time_key_cxr, x_cxr, x_cxr_mask)

        return proj_x_cxr_irg
    
    def forward(self, batch: Dict):
        #############################################
        # stage 1: encode both modalities
        #############################################
        # (batch_size, tt_max, embed_dim)
        proj_x_ts_irg = self.forward_ts_mtand(
            batch["ts"], batch["ts_mask"], batch["ts_tt"])
        cxr_imgs = batch["cxr_imgs"]
        cxr_time = batch["cxr_time"]
        cxr_time_mask = batch["cxr_time_mask"]
        batch_size = cxr_imgs.size(0)
        valid_cxr_imgs = cxr_imgs[cxr_time_mask.bool()]
        with torch.no_grad():
            cxr_feats = self.img_encoder(valid_cxr_imgs).img_embedding
        cxr_feats = self.img_proj_layer(cxr_feats)
        padded_feats = torch.zeros(
            batch_size, self.num_imgs, cxr_feats.size(-1)).type_as(cxr_feats)
        padded_feats[cxr_time_mask.bool()] = cxr_feats
        cxr_time_mask = cxr_time_mask.unsqueeze(2).repeat(1, 1, cxr_feats.size(-1))
        # (batch_size, tt_max, embed_dim)
        proj_x_cxr_irg = self.forward_cxr_mtand(
            padded_feats, cxr_time_mask, cxr_time)
        ts2vec_loss = self.hierarchical_contrastive_loss(
            proj_x_ts_irg, proj_x_cxr_irg)
        ipdb.set_trace()
        return ts2vec_loss

    # loss function of https://github.com/zhihanyue/ts2vec/tree/main
    def hierarchical_contrastive_loss(self, z1, z2, alpha=0.5, temporal_unit=0):
        loss = torch.tensor(0., device=z1.device)
        d = 0
        while z1.size(1) > 1:
            if alpha != 0:
                loss += alpha * self.instance_contrastive_loss(z1, z2)
            if d >= temporal_unit:
                if 1 - alpha != 0:
                    loss += (1 - alpha) * self.temporal_contrastive_loss(z1, z2)
            d += 1
            z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
            z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
        if z1.size(1) == 1:
            if alpha != 0:
                loss += alpha * self.instance_contrastive_loss(z1, z2)
            d += 1
        return loss / d

    @staticmethod
    def instance_contrastive_loss(z1, z2):
        B, T = z1.size(0), z1.size(1)
        if B == 1:
            return z1.new_tensor(0.)
        z = torch.cat([z1, z2], dim=0)  # 2B x T x C
        z = z.transpose(0, 1)  # T x 2B x C
        sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
        logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # T x 2B x (2B-1)
        logits += torch.triu(sim, diagonal=1)[:, :, 1:]
        logits = -F.log_softmax(logits, dim=-1)
        
        i = torch.arange(B, device=z1.device)
        loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
        return loss

    @staticmethod
    def temporal_contrastive_loss(z1, z2):
        B, T = z1.size(0), z1.size(1)
        if T == 1:
            return z1.new_tensor(0.)
        z = torch.cat([z1, z2], dim=1)  # B x 2T x C
        sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
        logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # B x 2T x (2T-1)
        logits += torch.triu(sim, diagonal=1)[:, :, 1:]
        logits = -F.log_softmax(logits, dim=-1)
        
        t = torch.arange(T, device=z1.device)
        loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
        return loss

    def training_step(self, batch: Dict, batch_idx: int):
        loss = self(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, 
                 prog_bar=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch: Dict, batch_idx: int):
        loss = self(batch)
        self.log("val_loss", loss, on_step=True, on_epoch=True, 
                 prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.ts_learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.4, patience=3, verbose=True, mode='min')
        scheduler = {
            'scheduler': lr_scheduler,
            'monitor': 'val_loss',
            'interval': 'epoch',
            'frequency': 1
        }
        return [optimizer], [scheduler]
    

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from cmehr.paths import *
    # from cmehr.dataset.mimic4_datamodule import MIMIC4DataModule
    from cmehr.dataset.mimic4_datamodule import MIMIC4DataModule

    datamodule = MIMIC4DataModule(
        file_path=str(ROOT_PATH / "output_mimic4/TS_CXR/ihm"),
        tt_max=48
    )
    batch = dict()
    for batch in datamodule.val_dataloader():
        break
    for k, v in batch.items():
        print(f"{k}: ", v.shape)
    """
    ts: torch.Size([4, 157, 17])
    ts_mask:  torch.Size([4, 157, 17])
    ts_tt:  torch.Size([4, 157])
    reg_ts:  torch.Size([4, 48, 34])
    input_ids:  torch.Size([4, 5, 128])
    attention_mask:  torch.Size([4, 5, 128])
    note_time:  torch.Size([4, 5])
    note_time_mask: torch.Size([4, 5])
    label: torch.Size([4])
    """
    model = MIMIC4PretrainModule(
        period_length=48)
    loss = model(batch)
    print(loss)
