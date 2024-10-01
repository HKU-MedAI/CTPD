from typing import Dict
import ipdb
import math
import numpy as np
from einops import rearrange
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from lightning import LightningModule
from timm.models.layers import DropPath
from cmehr.models.mimic4.UTDE_modules import multiTimeAttention
from cmehr.models.mimic4.tslanet_model import PatchEmbed, ICB
from cmehr.backbone.vision.pretrained import get_biovil_t_image_encoder
from cmehr.utils.hard_ts_losses import hier_CL_hard
from cmehr.utils.soft_ts_losses import hier_CL_soft
from cmehr.utils.lr_scheduler import linear_warmup_decay
from cmehr.models.common.dilated_conv import DilatedConvEncoder


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``0
        """
        return self.pe[:x.size(1)].permute(1, 0, 2)
    

class MIMIC4PretrainModule(LightningModule):
    def __init__(self,
                 orig_d_ts: int = 15,
                 orig_reg_d_ts: int = 30,
                 warmup_epochs: int = 20,
                 max_epochs: int = 100,
                 ts_learning_rate: float = 4e-4,
                 embed_time: int = 64,
                 embed_dim: int = 128,
                 num_imgs: int = 5,
                 period_length: float = 100,
                 cm_loss_weight: float = 0.,
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
        self.ts_learning_rate = ts_learning_rate
        self.embed_dim = embed_dim
        self.num_imgs = num_imgs
        self.tt_max = period_length
        self.cm_loss_weight = cm_loss_weight
        self.warmup_epochs = warmup_epochs

        self.img_encoder = get_biovil_t_image_encoder()
        # Don't freeze image encoder
        # for param in self.img_encoder.parameters():
        #     param.requires_grad = False
        self.img_embed_dim = 512
        self.img_proj_layer = nn.Linear(self.img_embed_dim, self.embed_dim)

        '''
        change this into two mtand:
        - TS mtand: 48 time points
        - CXR mtand: 5 time points
        '''

        self.ts_conv1 = nn.Conv1d(self.orig_d_ts, self.embed_dim, kernel_size=1)
        self.img_conv1 = nn.Conv1d(self.embed_dim, self.embed_dim, kernel_size=1)

        # object-centric learning ... 
        # do we still need pretraing?
        depth = 1
        self.ts_dilated_conv = DilatedConvEncoder(
            in_channels=self.embed_dim, 
            channels=[self.embed_dim] * depth + [self.embed_dim], 
            kernel_size=3
        )

        # step 1: mtand to project different timestamps
        # step 2: slot attention to learn dense slots between both modalities ... across timestamps
        # step 3: reconstruction within the latent space ?
        # step 4: contrastive learning within multiple scales. 

        # Add multiscale prototype ...
        # self.img_dilated_conv = DilatedConvEncoder(
        #     in_channels=self.embed_dim, 
        #     channels=[self.embed_dim] * depth + [self.embed_dim], 
        #     kernel_size=3
        # )
        # self.ts_patch_embed = PatchEmbed(
        #         seq_len=self.tt_max, patch_size=1,
        #         in_chans=self.embed_dim, embed_dim=self.embed_dim
        #     )
        # self.ts_icb = ICB(in_features=self.embed_dim, 
        #                  hidden_features=int(3 * self.embed_dim), 
        #                  drop=0.)
        # self.ts_pos_embed = PositionalEncoding(self.embed_dim)
        # self.ts_pos_drop = nn.Dropout(p=0.15)
        # self.ts_norm = nn.LayerNorm(self.embed_dim)
        # drop_path = 0.1
        # self.ts_drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # self.img_patch_embed = PatchEmbed(
        #         seq_len=self.tt_max, patch_size=1,
        #         in_chans=self.embed_dim, embed_dim=self.embed_dim
        #     )
        # self.img_icb = ICB(in_features=self.embed_dim, 
        #                hidden_features=int(3 * self.embed_dim), 
        #                drop=0.)
        # self.img_pos_embed = PositionalEncoding(self.embed_dim)
        # self.img_pos_drop = nn.Dropout(p=0.15)
        # self.img_norm = nn.LayerNorm(self.embed_dim)
        # drop_path = 0.1
        # self.img_drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # # for time series embedding
        # self.periodic_ts = nn.Linear(1, embed_time-1)
        # self.linear_ts = nn.Linear(1, 1)
        # # For TS, we encode it into hourly embedding within 400 hours ...
        # self.time_query_ts = torch.linspace(0, 1., self.tt_max)
        # self.time_attn_ts = multiTimeAttention(
        #     self.orig_d_ts*2, self.embed_dim, embed_time, 8)

        # # for img embedding
        # self.periodic_img = nn.Linear(1, embed_time-1)
        # self.linear_img = nn.Linear(1, 1)
        # # For CXR, we encode it into 5 time points ...
        # self.time_query_img = torch.linspace(0, 1., self.tt_max // 20)
        # self.time_attn_img = multiTimeAttention(
        #     self.embed_dim, self.embed_dim, embed_time, 8)
    
        self.train_iters_per_epoch = -1
        
    # def forward_ts_mtand(self,
    #                      x_ts: torch.Tensor,
    #                      x_ts_mask: torch.Tensor,
    #                      ts_tt_list: torch.Tensor):
    #     '''
    #     Forward irregular time series using mTAND.
    #     '''
    #     def learn_time_embedding(tt):
    #         tt = tt.unsqueeze(-1)
    #         out2 = torch.sin(self.periodic_ts(tt))
    #         out1 = self.linear_ts(tt)
    #         return torch.cat([out1, out2], -1)
    
    #     # (B, N) -> (B, N, embed_time)
    #     time_key_ts = learn_time_embedding(
    #         ts_tt_list)
    #     x_ts_irg = torch.cat((x_ts, x_ts_mask), 2)
    #     x_ts_mask = torch.cat((x_ts_mask, x_ts_mask), 2)

    #     time_query = learn_time_embedding(
    #         self.time_query_ts.unsqueeze(0).type_as(x_ts))
    #     # query: (1, N_r, embed_time),
    #     # key: (B, N, embed_time),
    #     # value: (B, N, 2 * D_t)
    #     # mask: (B, N, 2 * D_t)
    #     # out: (B, N_r, 128?)
    #     proj_x_ts_irg = self.time_attn_ts(
    #         time_query, time_key_ts, x_ts_irg, x_ts_mask)

    #     return proj_x_ts_irg

    # def forward_cxr_mtand(self,
    #                      x_cxr: torch.Tensor,
    #                      x_cxr_mask: torch.Tensor,
    #                      cxr_tt_list: torch.Tensor):
    #     '''
    #     Forward irregular CXRs using mTAND.
    #     '''
    #     def learn_time_embedding(tt):
    #         tt = tt.unsqueeze(-1)
    #         out2 = torch.sin(self.periodic_img(tt))
    #         out1 = self.linear_img(tt)
    #         return torch.cat([out1, out2], -1)
        
    #     time_key_cxr = learn_time_embedding(
    #         cxr_tt_list)
    #     time_query = learn_time_embedding(
    #         self.time_query_img.unsqueeze(0).type_as(x_cxr))
    #     proj_x_img_irg = self.time_attn_img(
    #         time_query, time_key_cxr, x_cxr, x_cxr_mask)

    #     return proj_x_img_irg

    @staticmethod
    def take_per_row(A, indx, num_elem):
        all_indx = indx[:,None] + np.arange(num_elem)
        return A[torch.arange(all_indx.shape[0])[:, None], all_indx]

    def aug_reg_ts(self, x):
        ''' Create two augmented views for regular time series'''
        ts_l = x.size(1)
        crop_l = np.random.randint(low=ts_l // 4, high=ts_l+1)
        crop_left = np.random.randint(ts_l - crop_l + 1)
        crop_right = crop_left + crop_l
        crop_eleft = np.random.randint(crop_left + 1)
        crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
        crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0))
        # crop_eleft < crop_left < crop_right < crop_eright
        # print(crop_eleft, crop_left, crop_right, crop_eright)
        x_aug_1 = self.take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft)
        x_aug_2 = self.take_per_row(x, crop_offset + crop_left, crop_eright - crop_left)

        return x_aug_1, x_aug_2, crop_l

    # def aug_irg_ts(self, ts, ts_tt, ts_mask):
    #     np.random.seed(42)
    #     # TODO: it denotes the minimum length of time series in a batch
    #     ts_l = torch.sum(ts_mask.sum(dim=2) != 0, dim=1).min().item()
    #     ts_l_max = ts_tt.size(1)
    #     # max sure the crop length is at least not all zero
    #     crop_l = np.random.randint(low=ts_l_max - ts_l + 1, high=ts_l_max+1)
    #     crop_left = np.random.randint(ts_l_max - crop_l + 1)
    #     crop_right = crop_left + crop_l
    #     crop_eleft = np.random.randint(crop_left + 1)
    #     crop_eright = np.random.randint(low=crop_right, high=ts_l_max + 1)
    #     crop_offset = np.random.randint(low=-crop_eleft, high=ts_l_max - crop_eright + 1, size=ts.size(0))
    #     ts_aug_1 = self.take_per_row(ts, crop_offset + crop_eleft, crop_right - crop_eleft)
    #     ts_tt_aug_1 = self.take_per_row(ts_tt, crop_offset + crop_eleft, crop_right - crop_eleft)
    #     ts_mask_aug_1 = self.take_per_row(ts_mask, crop_offset + crop_eleft, crop_right - crop_eleft)
    #     ts_aug_2 = self.take_per_row(ts, crop_offset + crop_left, crop_eright - crop_left)
    #     ts_tt_aug_2 = self.take_per_row(ts_tt, crop_offset + crop_left, crop_eright - crop_left)
    #     ts_mask_aug_2 = self.take_per_row(ts_mask, crop_offset + crop_left, crop_eright - crop_left)

    #     return ts_aug_1, ts_tt_aug_1, ts_mask_aug_1, ts_aug_2, ts_tt_aug_2, ts_mask_aug_2

    # def extract_img_embs(self, imgs, img_time, img_time_mask):
    #     batch_size = imgs.size(0)
    #     valid_imgs = imgs[img_time_mask.bool()]
    #     cxr_feats = self.img_encoder(valid_imgs).img_embedding
    #     cxr_embs = self.img_proj_layer(cxr_feats)

    #     padded_feats = torch.zeros(
    #         batch_size, self.num_imgs, cxr_embs.size(-1)).type_as(cxr_embs)
    #     padded_feats[img_time_mask.bool()] = cxr_embs
    #     img_time_mask = img_time_mask.unsqueeze(2).repeat(1, 1, cxr_embs.size(-1))
    #     proj_img_embs = self.forward_cxr_mtand(
    #         padded_feats, img_time_mask, img_time)
        
    #     return proj_img_embs

    def forward(self, batch: Dict):
        # TODO: consider the reg_ts in the frequency domain ...
        batch_size = batch["ts"].size(0)
        reg_ts = batch["reg_ts"][..., :self.orig_d_ts]
        # reg_ts = batch["reg_ts"]

        # create two augmentation view for regular TS
        ts_aug_1, ts_aug_2, crop_l = self.aug_reg_ts(reg_ts)

        # Embedding for each individual timestamp
        feat_ts_aug_1 = self.ts_conv1(ts_aug_1.permute(0, 2, 1))
        feat_ts_aug_2 = self.ts_conv1(ts_aug_2.permute(0, 2, 1))

        # Learn temporal interaction
        emb_ts_aug_1 = self.ts_dilated_conv(feat_ts_aug_1).permute(0, 2, 1)
        emb_ts_aug_1 = F.normalize(emb_ts_aug_1, dim=-1)
        emb_ts_aug_2 = self.ts_dilated_conv(feat_ts_aug_2).permute(0, 2, 1)
        emb_ts_aug_2 = F.normalize(emb_ts_aug_2, dim=-1)

        emb_ts_aug_1 = emb_ts_aug_1[:, -crop_l:]
        emb_ts_aug_2 = emb_ts_aug_2[:, :crop_l]
        ts2vec_loss = hier_CL_hard(
            emb_ts_aug_1, emb_ts_aug_2
        )

        if torch.isnan(ts2vec_loss):
            from cmehr.utils.hard_ts_losses import inst_CL_hard, temp_CL_hard
            ipdb.set_trace()

        # (batch_size, tt_max, embed_dim)
        # ts, ts_mask, ts_tt = batch["ts"], batch["ts_mask"], batch["ts_tt"]
        # create two augmentation view for TS
        # ts_aug_1, ts_tt_aug_1, ts_mask_aug_1, ts_aug_2, ts_tt_aug_2, ts_mask_aug_2 = self.aug_ts(ts, ts_tt, ts_mask)
        # proj_ts_aug_1 = self.forward_ts_mtand(
        #     ts_aug_1, ts_mask_aug_1, ts_tt_aug_1)
        # proj_ts_aug_2 = self.forward_ts_mtand(
        #     ts_aug_2, ts_mask_aug_2, ts_tt_aug_2)

        # Create image embeddings
        reg_imgs = rearrange(batch["reg_imgs"], "b n c h w -> (b n) c h w")
        cxr_feats = self.img_encoder(reg_imgs).img_embedding
        cxr_embs = self.img_proj_layer(cxr_feats)
        cxr_embs = rearrange(cxr_embs, "(b n) d -> b n d", b=batch_size)
        # cxr_mask = batch["reg_imgs_mask"].unsqueeze(-1).repeat(1, 1, cxr_embs.size(-1))
        # cxr_embs = torch.cat((cxr_embs, cxr_mask), 2)
        img_emb = self.img_conv1(cxr_embs.permute(0, 2, 1)).permute(0, 2, 1)
        # img_emb = self.img_dilated_conv(img_feat)
        num_imgs = img_emb.size(1)

        # Cross modal loss
        feat_ts = self.ts_conv1(reg_ts.permute(0, 2, 1))
        emb_ts = self.ts_dilated_conv(feat_ts)
        avg_ts_emb = F.avg_pool1d(emb_ts, kernel_size=(self.tt_max // num_imgs), 
                                  stride=(self.tt_max // num_imgs))
        avg_ts_emb = avg_ts_emb.permute(0, 2, 1)
        avg_ts_emb = rearrange(avg_ts_emb, "b n d -> (b n) d")
        img_emb = rearrange(img_emb, "b n d -> (b n) d")
        cm_loss = self.infonce_loss(avg_ts_emb, img_emb)

        loss_dict = {
            "loss": ts2vec_loss + self.cm_loss_weight * cm_loss,
            "ts2vec_loss": ts2vec_loss,
            "cm_loss": cm_loss
        }
        return loss_dict

    def infonce_loss(self, out_1, out_2, temperature=0.07):
        """
        Compute the InfoNCE loss for the given outputs.
        """
        out_1 = F.normalize(out_1, dim=-1)
        out_2 = F.normalize(out_2, dim=-1)
        sim = torch.matmul(out_1, out_2.transpose(0, 1))
        sim /= temperature
        labels = torch.arange(sim.size(0)).to(sim.device)
        return F.cross_entropy(sim, labels)

    def training_step(self, batch: Dict, batch_idx: int):
        batch_size = batch["ts"].size(0)
        loss_dict = self(batch)
        train_loss_dict = {f"train_{k}": v for k, v in loss_dict.items()}
        self.log_dict(train_loss_dict, on_step=True, on_epoch=True, 
                      prog_bar=True, sync_dist=True, batch_size=batch_size)
        return loss_dict["loss"]
    
    def validation_step(self, batch: Dict, batch_idx: int):
        batch_size = batch["ts"].size(0)
        loss_dict = self(batch)
        val_loss_dict = {f"val_{k}": v for k, v in loss_dict.items()}
        self.log_dict(val_loss_dict, on_step=True, on_epoch=True, 
                      prog_bar=True, sync_dist=True, batch_size=batch_size)
        return loss_dict["loss"]

    def configure_optimizers(self):
        optimizer= torch.optim.Adam([
                {'params': [p for n, p in self.named_parameters() if 'img_encoder' not in n]},
                {'params':[p for n, p in self.named_parameters() if 'img_encoder' in n], 'lr': self.ts_learning_rate / 10}
            ], lr=self.ts_learning_rate)
        
        assert self.train_iters_per_epoch != -1, "train_iters_per_epoch is not set"
        warmup_steps = self.train_iters_per_epoch * self.warmup_epochs
        total_steps = self.train_iters_per_epoch * self.max_epochs

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                linear_warmup_decay(warmup_steps, total_steps, cosine=True),
            ),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]
    

if __name__ == "__main__":
    from cmehr.paths import *
    # from cmehr.dataset.mimic4_datamodule import MIMIC4DataModule
    from cmehr.dataset.mimic4_pretraining_datamodule import MIMIC4MultimodalDataModule

    datamodule = MIMIC4MultimodalDataModule(
        file_path=str(ROOT_PATH / "output_mimic4/self_supervised_multimodal"),
        period_length=48
    )
    batch = dict()
    for batch in datamodule.val_dataloader():
        break
    for k, v in batch.items():
        print(f"{k}: ", v.shape)

    model = MIMIC4PretrainModule(period_length=48, num_imgs=4)
    loss = model(batch)
    print(loss)
