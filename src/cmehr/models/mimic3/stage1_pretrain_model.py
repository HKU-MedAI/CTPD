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
from transformers import AutoModel
from timm.models.layers import DropPath
from cmehr.models.mimic4.UTDE_modules import multiTimeAttention
from cmehr.models.mimic4.tslanet_model import PatchEmbed, ICB
from cmehr.utils.hard_ts_losses import hier_CL_hard
from cmehr.utils.soft_ts_losses import hier_CL_soft
from cmehr.utils.lr_scheduler import linear_warmup_decay
from cmehr.models.common.dilated_conv import DilatedConvEncoder
from cmehr.models.mimic3.bert_modules import BertForRepresentation


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
    

class MIMIC3PretrainModule(LightningModule):
    def __init__(self,
                 orig_d_ts: int = 17,
                 orig_reg_d_ts: int = 34,
                 warmup_epochs: int = 20,
                 max_epochs: int = 100,
                 ts_learning_rate: float = 4e-4,
                 embed_time: int = 64,
                 embed_dim: int = 128,
                 num_imgs: int = 5,
                 period_length: float = 100,
                 cm_loss_weight: float = 0.,
                 text_model_name: str = "yikuan8/Clinical-Longformer",
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

        # need to use BERT here ...
        Biobert = AutoModel.from_pretrained(text_model_name)
        self.bertrep = BertForRepresentation(text_model_name, Biobert)

        # can't train this model when bert are trainable
        # for param in self.bertrep.parameters():
        #     param.requires_grad = False
        
        '''
        change this into two mtand:
        - TS mtand: 48 time points
        - CXR mtand: 5 time points
        '''

        self.ts_conv1 = nn.Conv1d(self.orig_d_ts, self.embed_dim, kernel_size=1)
        self.text_conv1 = nn.Conv1d(768, self.embed_dim, kernel_size=1)

        depth = 1
        self.ts_dilated_conv = DilatedConvEncoder(
            in_channels=self.embed_dim, 
            channels=[self.embed_dim] * depth + [self.embed_dim], 
            kernel_size=3
        )

        # self.text_dilated_conv = DilatedConvEncoder(
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

    def forward(self, batch: Dict):
        # TODO: consider the reg_ts in the frequency domain ...
        batch_size = batch["ts"].size(0)
        reg_ts = batch["reg_ts"][..., :self.orig_d_ts]

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

        # if torch.isnan(ts2vec_loss):
        #     from cmehr.utils.hard_ts_losses import inst_CL_hard, temp_CL_hard
        #     ipdb.set_trace()

        # # (batch_size, tt_max, embed_dim)
        # # ts, ts_mask, ts_tt = batch["ts"], batch["ts_mask"], batch["ts_tt"]
        # # create two augmentation view for TS
        # # ts_aug_1, ts_tt_aug_1, ts_mask_aug_1, ts_aug_2, ts_tt_aug_2, ts_mask_aug_2 = self.aug_ts(ts, ts_tt, ts_mask)
        # # proj_ts_aug_1 = self.forward_ts_mtand(
        # #     ts_aug_1, ts_mask_aug_1, ts_tt_aug_1)
        # # proj_ts_aug_2 = self.forward_ts_mtand(
        # #     ts_aug_2, ts_mask_aug_2, ts_tt_aug_2)

        # Create text embeddings
        # TODO: here two approaches to consider:
        # 1. use the original text embedding
        # 2. use the interpolated text embedding
        x_txt = self.bertrep(batch["input_ids"], batch["attention_mask"]) # x_txt: bz, 5, 768
        text_emb = self.text_conv1(x_txt.permute(0, 2, 1)).permute(0, 2, 1)
        num_notes = text_emb.size(1)

        # Cross modal loss
        feat_ts = self.ts_conv1(reg_ts.permute(0, 2, 1))
        emb_ts = self.ts_dilated_conv(feat_ts)
        avg_ts_emb = F.avg_pool1d(emb_ts, kernel_size=(self.tt_max // num_notes), 
                                  stride=(self.tt_max // num_notes))
        avg_ts_emb = avg_ts_emb.permute(0, 2, 1)
        avg_ts_emb = rearrange(avg_ts_emb, "b n d -> (b n) d")
        text_emb = rearrange(text_emb, "b n d -> (b n) d")
        cm_loss = self.infonce_loss(avg_ts_emb, text_emb)

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
                {'params': [p for n, p in self.named_parameters() if 'bert' not in n]},
                {'params':[p for n, p in self.named_parameters() if 'bert' in n], 'lr': self.ts_learning_rate / 10}
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
    