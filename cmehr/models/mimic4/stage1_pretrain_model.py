from typing import Dict
import ipdb
import numpy as np
import sklearn.metrics as metrics
from einops import rearrange
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

        '''
        change this into two mtand:
        - TS mtand: 48 time points
        - CXR mtand: 5 time points
        '''
        # for time series embedding
        self.periodic_ts = nn.Linear(1, embed_time-1)
        self.linear_ts = nn.Linear(1, 1)
        self.time_query_ts = torch.linspace(0, 1., self.tt_max)
        self.time_attn_ts = multiTimeAttention(
            self.orig_d_ts*2, self.embed_dim, embed_time, 8)
        self.lstm_ts = nn.LSTM(
            embed_dim, embed_dim, 1, batch_first=True)

        # for img embedding
        self.periodic_img = nn.Linear(1, embed_time-1)
        self.linear_img = nn.Linear(1, 1)
        self.time_query_img = torch.linspace(0, 1., self.num_imgs)
        self.time_attn_img = multiTimeAttention(
            self.embed_dim, self.embed_dim, embed_time, 8)
        self.lstm_img = nn.LSTM(
            embed_dim, embed_dim, 1, batch_first=True)

    def forward_ts_mtand(self,
                         x_ts: torch.Tensor,
                         x_ts_mask: torch.Tensor,
                         ts_tt_list: torch.Tensor):
        '''
        Forward irregular time series using mTAND.
        '''
        def learn_time_embedding(tt):
            tt = tt.unsqueeze(-1)
            out2 = torch.sin(self.periodic_ts(tt))
            out1 = self.linear_ts(tt)
            return torch.cat([out1, out2], -1)
    
        # (B, N) -> (B, N, embed_time)
        time_key_ts = learn_time_embedding(
            ts_tt_list)
        x_ts_irg = torch.cat((x_ts, x_ts_mask), 2)
        x_ts_mask = torch.cat((x_ts_mask, x_ts_mask), 2)

        time_query = learn_time_embedding(
            self.time_query_ts.unsqueeze(0).type_as(x_ts))
        # query: (1, N_r, embed_time),
        # key: (B, N, embed_time),
        # value: (B, N, 2 * D_t)
        # mask: (B, N, 2 * D_t)
        # out: (B, N_r, 128?)
        proj_x_ts_irg = self.time_attn_ts(
            time_query, time_key_ts, x_ts_irg, x_ts_mask)
        
        proj_x_ts_irg, _ = self.lstm_ts(proj_x_ts_irg)

        return proj_x_ts_irg

    def forward_cxr_mtand(self,
                         x_cxr: torch.Tensor,
                         x_cxr_mask: torch.Tensor,
                         cxr_tt_list: torch.Tensor):
        '''
        Forward irregular CXRs using mTAND.
        '''
        def learn_time_embedding(tt):
            tt = tt.unsqueeze(-1)
            out2 = torch.sin(self.periodic_img(tt))
            out1 = self.linear_img(tt)
            return torch.cat([out1, out2], -1)
        
        # (B, N) -> (B, N, embed_time)
        # fixed
        time_key_cxr = learn_time_embedding(
            cxr_tt_list)
        time_query = learn_time_embedding(
            self.time_query_img.unsqueeze(0).type_as(x_cxr))
        # query: (1, N_r, embed_time),
        # key: (B, N, embed_time),
        # value: (B, N, 2 * D_t)
        # mask: (B, N, 2 * D_t)
        # out: (B, N_r, 128?)
        proj_x_img_irg = self.time_attn_img(
            time_query, time_key_cxr, x_cxr, x_cxr_mask)
        
        proj_x_img_irg, _ = self.lstm_img(proj_x_img_irg)

        return proj_x_img_irg

    @staticmethod
    def take_per_row(A, indx, num_elem):
        all_indx = indx[:,None] + np.arange(num_elem)
        return A[torch.arange(all_indx.shape[0])[:, None], all_indx]

    def aug_ts(self, ts, ts_tt, ts_mask):
        np.random.seed(42)
        # TODO: it denotes the minimum length of time series in a batch
        ts_l = torch.sum(ts_mask.sum(dim=2) != 0, dim=1).min().item()
        ts_l_max = ts_tt.size(1)
        # max sure the crop length is at least not all zero
        crop_l = np.random.randint(low=ts_l_max - ts_l + 1, high=ts_l_max+1)
        crop_left = np.random.randint(ts_l_max - crop_l + 1)
        crop_right = crop_left + crop_l
        crop_eleft = np.random.randint(crop_left + 1)
        crop_eright = np.random.randint(low=crop_right, high=ts_l_max + 1)
        crop_offset = np.random.randint(low=-crop_eleft, high=ts_l_max - crop_eright + 1, size=ts.size(0))
        ts_aug_1 = self.take_per_row(ts, crop_offset + crop_eleft, crop_right - crop_eleft)
        ts_tt_aug_1 = self.take_per_row(ts_tt, crop_offset + crop_eleft, crop_right - crop_eleft)
        ts_mask_aug_1 = self.take_per_row(ts_mask, crop_offset + crop_eleft, crop_right - crop_eleft)
        ts_aug_2 = self.take_per_row(ts, crop_offset + crop_left, crop_eright - crop_left)
        ts_tt_aug_2 = self.take_per_row(ts_tt, crop_offset + crop_left, crop_eright - crop_left)
        ts_mask_aug_2 = self.take_per_row(ts_mask, crop_offset + crop_left, crop_eright - crop_left)

        return ts_aug_1, ts_tt_aug_1, ts_mask_aug_1, ts_aug_2, ts_tt_aug_2, ts_mask_aug_2

    def forward(self, batch: Dict):
        #############################################
        # stage 1: encode both modalities
        #############################################
        # (batch_size, tt_max, embed_dim)
        ts, ts_mask, ts_tt = batch["ts"], batch["ts_mask"], batch["ts_tt"]
        # create two augmentation view for TS
        ts_aug_1, ts_tt_aug_1, ts_mask_aug_1, ts_aug_2, ts_tt_aug_2, ts_mask_aug_2 = self.aug_ts(ts, ts_tt, ts_mask)
        proj_ts_aug_1 = self.forward_ts_mtand(
            ts_aug_1, ts_mask_aug_1, ts_tt_aug_1)
        proj_ts_aug_2 = self.forward_ts_mtand(
            ts_aug_2, ts_mask_aug_2, ts_tt_aug_2)

        cxr_imgs = batch["cxr_imgs"]
        cxr_time = batch["cxr_time"] 
        cxr_time_mask = batch["cxr_time_mask"]
        batch_size = cxr_imgs.size(0)
        valid_cxr_imgs = cxr_imgs[cxr_time_mask.bool()]
        cxr_feats = self.img_encoder(valid_cxr_imgs).img_embedding
        cxr_embs = self.img_proj_layer(cxr_feats)

        padded_feats = torch.zeros(
            batch_size, self.num_imgs, cxr_embs.size(-1)).type_as(cxr_embs)
        padded_feats[cxr_time_mask.bool()] = cxr_embs
        cxr_time_mask = cxr_time_mask.unsqueeze(2).repeat(1, 1, cxr_embs.size(-1))
        proj_img_embs = self.forward_cxr_mtand(
            padded_feats, cxr_time_mask, cxr_time)

        ts2vec_loss = self.hierarchical_contrastive_loss(
            proj_ts_aug_1, proj_ts_aug_2)

        # find corresponding CXR at each time point
        cxr_time_indices = self.time_query_img // (1 / self.tt_max)
        ts_embs_aug_1 = proj_ts_aug_1[torch.arange(batch_size)[:, None], cxr_time_indices.long()]
        ts_embs_aug_2 = proj_ts_aug_2[torch.arange(batch_size)[:, None], cxr_time_indices.long()]

        ts_embs_aug_1 = rearrange(ts_embs_aug_1, "b n d -> (b n) d")
        ts_embs_aug_2 = rearrange(ts_embs_aug_2, "b n d -> (b n) d")
        proj_img_embs = rearrange(proj_img_embs, "b n d -> (b n) d")
        cm_loss = (self.infonce_loss(ts_embs_aug_1, proj_img_embs) + self.infonce_loss(ts_embs_aug_2, proj_img_embs)) / 2
        loss_dict = {
            "loss": ts2vec_loss + cm_loss,
            "ts2vec_loss": ts2vec_loss,
            "cm_loss": cm_loss
        }
        return loss_dict

    def infonce_loss(self, out_1, out_2, temperature=0.5):
        """
        Compute the InfoNCE loss for the given outputs.
        """
        out_1 = F.normalize(out_1, dim=-1)
        out_2 = F.normalize(out_2, dim=-1)
        sim = torch.matmul(out_1, out_2.transpose(0, 1))
        sim /= temperature
        labels = torch.arange(sim.size(0)).to(sim.device)
        return F.cross_entropy(sim, labels)
    
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
        loss_dict = self(batch)
        train_loss_dict = {f"train_{k}": v for k, v in loss_dict.items()}
        self.log_dict(train_loss_dict, on_step=True, on_epoch=True, 
                      prog_bar=True, sync_dist=True)
        return loss_dict["loss"]
    
    def validation_step(self, batch: Dict, batch_idx: int):
        loss_dict = self(batch)
        val_loss_dict = {f"val_{k}": v for k, v in loss_dict.items()}
        self.log_dict(val_loss_dict, on_step=True, on_epoch=True, 
                      prog_bar=True, sync_dist=True)
        return loss_dict["loss"]

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
