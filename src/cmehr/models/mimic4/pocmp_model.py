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
from cmehr.models.mimic4.UTDE_modules import multiTimeAttention, gateMLP
# from cmehr.models.mimic4.tslanet_model import PatchEmbed, ICB
from cmehr.backbone.vision.pretrained import get_biovil_t_image_encoder
from cmehr.models.mimic4.base_model import MIMIC4LightningModule
from cmehr.models.mimic4.mtand_model import Attn_Net_Gated
# from cmehr.utils.hard_ts_losses import hier_CL_hard
# from cmehr.utils.soft_ts_losses import hier_CL_soft
from cmehr.utils.lr_scheduler import linear_warmup_decay
from cmehr.models.common.dilated_conv import DilatedConvEncoder, ConvBlock
from cmehr.models.mimic4.position_encode import PositionalEncoding1D


class SlotAttention(nn.Module):
    '''
    Implementation of original slot attention.
    '''

    def __init__(self, num_slots, dim, iters=3, eps=1e-8, hidden_dim=128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_sigma = nn.Parameter(torch.rand(1, 1, dim))

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs, num_slots=None):
        b, n, d = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots
        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_sigma.expand(b, n_s, -1)
        slots = torch.normal(mu, sigma)

        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum('bjd,bij->bid', v, attn)

            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )

            slots = slots.reshape(b, -1, d)
            slots = slots + self.fc2(F.relu(self.fc1(self.norm_pre_ff(slots))))

        return slots, attn
    

class POCMPModule(MIMIC4LightningModule):
    '''
    The class of prototype-oriented contrastive multi-modal pretraining model.
    '''
    def __init__(self,
                 task: str = "ihm",
                 orig_d_ts: int = 15,
                 orig_reg_d_ts: int = 30,
                 warmup_epochs: int = 20,
                 max_epochs: int = 100,
                 ts_learning_rate: float = 4e-4,
                 embed_time: int = 64,
                 embed_dim: int = 128,
                 num_imgs: int = 4,
                 period_length: float = 48,
                 lamb1: float = 1.,
                 lamb2: float = 2.,
                 use_prototype: bool = True,
                 use_multiscale: bool = True,
                 TS_mixup: bool = True,
                 mixup_level: str = "batch",
                 dropout: float = 0.1,
                 pooling_type: str = "attention",
                 *args,
                 **kwargs
                 ):
        ''' Maybe to extract visual features offline ...
        
        lamb1: controls the weight of the slot loss
        lamb2: controls the weight of the contrastive loss
        
        '''
        # TODO: add more arguments for ablation study
        super().__init__(task=task, max_epochs=max_epochs,
                         ts_learning_rate=ts_learning_rate,
                         period_length=period_length)
        self.save_hyperparameters()
        
        self.orig_d_ts = orig_d_ts      
        self.orig_reg_d_ts = orig_reg_d_ts
        self.max_epochs = max_epochs
        self.ts_learning_rate = ts_learning_rate
        self.embed_dim = embed_dim
        self.num_imgs = num_imgs
        # self.tt_max = period_length
        self.lamb1 = lamb1
        self.lamb2 = lamb2
        self.warmup_epochs = warmup_epochs
        self.use_prototype = use_prototype
        self.use_multiscale = use_multiscale
        self.pooling_type = pooling_type
        self.dropout = dropout

        self.img_encoder = get_biovil_t_image_encoder()
        self.img_embed_dim = 512
        self.img_proj_layer = nn.Linear(self.img_embed_dim, self.embed_dim)

        # define convolution within multiple layers
        self.ts_conv_1 = ConvBlock(
                self.embed_dim,
                self.embed_dim,
                kernel_size=3,
                dilation=1,
                final=False,
            )
        self.ts_conv_2 = ConvBlock(
                self.embed_dim,
                self.embed_dim,
                kernel_size=3,
                dilation=2,
                final=False,
            )
        self.ts_conv_3 = ConvBlock(
                self.embed_dim,
                self.embed_dim,
                kernel_size=3,
                dilation=4,
                final=True,
            )

        self.img_conv_1 = ConvBlock(
            self.embed_dim,
            self.embed_dim,
            kernel_size=3,
            dilation=1,
            final=False,
        )

        self.periodic = nn.Linear(1, embed_time-1)
        self.linear = nn.Linear(1, 1)
        self.time_query_ts = torch.linspace(0, 1., self.tt_max)
        self.time_query_img = torch.linspace(0, 1., self.num_imgs)
        self.time_attn_ts = multiTimeAttention(
            self.orig_d_ts*2, self.embed_dim, embed_time, 8)
        self.time_attn_img = multiTimeAttention(
            self.embed_dim, self.embed_dim, embed_time, 8)
        
        self.TS_mixup = TS_mixup
        self.mixup_level = mixup_level

        if self.TS_mixup:
            if self.mixup_level == 'batch':
                self.moe_ts = gateMLP(
                    input_dim=self.embed_dim*2, hidden_size=embed_dim, output_dim=1, dropout=dropout)
                self.moe_img = gateMLP(
                    input_dim=self.embed_dim*2, hidden_size=embed_dim, output_dim=1, dropout=dropout)
            # elif self.mixup_level == 'batch_seq':
            #     self.moe = gateMLP(
            #         input_dim=self.embed_dim*2, hidden_size=embed_dim, output_dim=1, dropout=dropout)
            # elif self.mixup_level == 'batch_seq_feature':
            #     self.moe = gateMLP(
            #         input_dim=self.embed_dim*2, hidden_size=embed_dim, output_dim=self.embed_dim, dropout=dropout)
            else:
                raise ValueError("Unknown mixedup type")

        self.proj_reg_ts = nn.Conv1d(orig_reg_d_ts, self.embed_dim, kernel_size=1, padding=0, bias=False)
        self.proj_reg_img = nn.Conv1d(self.embed_dim, self.embed_dim, kernel_size=1, padding=0, bias=False)

        if self.use_prototype:
            self.pe = PositionalEncoding1D(embed_dim)
            if self.use_multiscale:
                num_prototypes = [20, 10, 5]
            else:
                num_prototypes = [10]
            self.ts_grouping = []
            for i, num_prototype in enumerate(num_prototypes):
                self.ts_grouping.append(SlotAttention(
                    dim=embed_dim, num_slots=num_prototype))
            self.ts_grouping = nn.ModuleList(self.ts_grouping)
            self.img_grouping = SlotAttention(
                dim=embed_dim, num_slots=10)

        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4)
        self.fusion_layer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.ts_atten_pooling = Attn_Net_Gated(
            L=embed_dim, D=64, dropout=True, n_classes=1)
        self.img_atten_pooling = Attn_Net_Gated(
            L=embed_dim, D=64, dropout=True, n_classes=1)
        
        # because the input is a concatenation
        # self.proj1 = nn.Linear(2 * self.embed_dim, self.embed_dim)
        # self.proj2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_layer = nn.Linear(2 * self.embed_dim, self.num_labels)

        self.train_iters_per_epoch = -1

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

        x_ts_irg = torch.cat((x_ts, x_ts_mask), 2)
        x_ts_mask = torch.cat((x_ts_mask, x_ts_mask), 2)
        # (B, N) -> (B, N, embed_time)
        # fixed
        time_key_ts = self.learn_time_embedding(
            ts_tt_list)
        time_query = self.learn_time_embedding(
            self.time_query_ts.unsqueeze(0).type_as(x_ts))
        # query: (1, N_r, embed_time),
        # key: (B, N, embed_time),
        # value: (B, N, 2 * D_t)
        # mask: (B, N, 2 * D_t)
        # out: (B, N_r, 128?)
        proj_x_ts_irg = self.time_attn_ts(
            time_query, time_key_ts, x_ts_irg, x_ts_mask)
        proj_x_ts_irg = proj_x_ts_irg.transpose(0, 1)

        return proj_x_ts_irg
    
    def forward_ts_reg(self, reg_ts: torch.Tensor):
        '''
        Forward irregular time series using Imputation.
        '''
        # convolution over regular time series
        x_ts_reg = reg_ts.transpose(1, 2)
        proj_x_ts_reg = x_ts_reg if self.orig_reg_d_ts == self.d_ts else self.proj_ts(
            x_ts_reg)
        proj_x_ts_reg = proj_x_ts_reg.permute(2, 0, 1)

        return proj_x_ts_reg

    def gate_ts(self,
                proj_x_ts_irg: torch.Tensor,
                proj_x_ts_reg: torch.Tensor):

        assert self.TS_mixup, "TS_mixup is not enabled"
        if self.mixup_level == 'batch':
            g_irg = torch.max(proj_x_ts_irg, dim=0).values
            g_reg = torch.max(proj_x_ts_reg, dim=0).values
            moe_gate = torch.cat([g_irg, g_reg], dim=-1)
        elif self.mixup_level == 'batch_seq' or self.mixup_level == 'batch_seq_feature':
            moe_gate = torch.cat(
                [proj_x_ts_irg, proj_x_ts_reg], dim=-1)
        else:
            raise ValueError("Unknown mixup type")
        mixup_rate = self.moe_ts(moe_gate)
        proj_x_ts = mixup_rate * proj_x_ts_irg + \
            (1 - mixup_rate) * proj_x_ts_reg

        return proj_x_ts

    # def gate_img(self,
    #              proj_x_img_irg: torch.Tensor,
    #              proj_x_img_reg: torch.Tensor):
        
    #     assert self.TS_mixup, "TS_mixup is not enabled"
    #     if self.mixup_level == 'batch':
    #         g_irg = torch.max(proj_x_img_irg, dim=0).values
    #         g_reg = torch.max(proj_x_img_reg, dim=0).values
    #         moe_gate = torch.cat([g_irg, g_reg], dim=-1)
    #     elif self.mixup_level == 'batch_seq' or self.mixup_level == 'batch_seq_feature':
    #         moe_gate = torch.cat(
    #             [proj_x_img_irg, proj_x_img_reg], dim=-1)
    #     else:
    #         raise ValueError("Unknown mixup type")
    
    #     mixup_rate = self.moe_img(moe_gate)
    #     proj_x_img = mixup_rate * proj_x_img_irg + \
    #         (1 - mixup_rate) * proj_x_img_reg
        
    #     return proj_x_img
    
    def forward_img_mtand(self, 
                          cxr_imgs: torch.Tensor,
                          cxr_time: torch.Tensor,
                          cxr_time_mask: torch.Tensor):
        
        valid_imgs = cxr_imgs[cxr_time_mask.bool()]
        x_img = self.img_encoder(valid_imgs).img_embedding
        x_img = self.img_proj_layer(x_img)
        B, N, _, _, _ = cxr_imgs.size()
        pad_x_img = torch.zeros(B, N, self.embed_dim).type_as(x_img)
        pad_x_img[cxr_time_mask.bool()] = x_img
        x_img = pad_x_img

        time_key_img = self.learn_time_embedding(
            cxr_time)
        time_query = self.learn_time_embedding(
            self.time_query_img.unsqueeze(0).type_as(x_img))
        mask_img = cxr_time_mask.unsqueeze(-1).expand(-1, -1, self.embed_dim)
        proj_x_img_irg = self.time_attn_img(
            time_query, time_key_img, x_img, mask_img)
        proj_x_img_irg = proj_x_img_irg.transpose(0, 1)

        return proj_x_img_irg

    def forward_ts_reg(self, reg_ts: torch.Tensor):
        '''
        Forward regular time series.
        '''
        # convolution over regular time series
        x_ts_reg = reg_ts.transpose(1, 2)
        proj_x_ts_reg = self.proj_reg_ts(x_ts_reg)
        proj_x_ts_reg = proj_x_ts_reg.permute(2, 0, 1)

        return proj_x_ts_reg
    
    # def forward_img_reg(self, reg_imgs: torch.Tensor, reg_imgs_mask: torch.Tensor):
    #     batch_size = reg_imgs.size(0)
    #     reg_imgs = rearrange(reg_imgs, 'b n c h w -> (b n) c h w')
    #     x_img = self.img_encoder(reg_imgs).img_embedding
    #     x_img = self.img_proj_layer(x_img)
    #     x_img = rearrange(x_img, '(b n) c -> b n c', b=batch_size)
    #     x_img = x_img.transpose(1, 2)
    #     proj_x_img_reg = self.proj_reg_img(x_img)
    #     proj_x_img_reg = proj_x_img_reg.permute(2, 0, 1)

    #     return proj_x_img_reg

    def infonce_loss(self, out_1, out_2, temperature=0.07):
        """
        Compute the InfoNCE loss for the given outputs.
        """
        out_1 = F.normalize(out_1, dim=-1)
        out_2 = F.normalize(out_2, dim=-1)
        sim = torch.matmul(out_1, out_2.transpose(0, 1))
        sim /= temperature
        labels = torch.arange(sim.size(0)).type_as(sim).long()
        return F.cross_entropy(sim, labels)

    def forward(self, x_ts, x_ts_mask, ts_tt_list,
                cxr_imgs, cxr_time, cxr_time_mask,
                reg_imgs, reg_imgs_mask, reg_ts,
                labels=None):
        
        # STEP 1: extract embeddings from irregular data.
        proj_x_ts_irg = self.forward_ts_mtand(
            x_ts, x_ts_mask, ts_tt_list)
        proj_x_ts_reg = self.forward_ts_reg(reg_ts)
        proj_x_ts = self.gate_ts(proj_x_ts_irg, proj_x_ts_reg)
        proj_x_ts = rearrange(proj_x_ts, "tt b d -> b d tt")
        proj_x_img_irg = self.forward_img_mtand(
            cxr_imgs, cxr_time, cxr_time_mask)
        # no need to mixup for image.
        # proj_x_img_reg = self.forward_img_reg(reg_imgs, reg_imgs_mask)
        # proj_x_img = self.gete_img(proj_x_img_irg, proj_x_img_reg)
        proj_x_img = rearrange(proj_x_img_irg, "tt b d -> b d tt")
        # proj_x_img = rearrange(proj_x_img, "tt b d -> b d tt")

        # STEP 2: multi-scale features
        ts_emb_1 = self.ts_conv_1(proj_x_ts)
        ts_emb_2 = F.avg_pool1d(ts_emb_1, 2)
        ts_emb_2 = self.ts_conv_2(ts_emb_2)
        ts_emb_3 = F.avg_pool1d(ts_emb_2, 2)
        ts_emb_3 = self.ts_conv_3(ts_emb_3)

        # STEP 3: extract prototypes from the multi-scale features
        slot_loss = 0.
        ts_slot_list = []
        ts_feat_list = []
        # STEP 3: prototype-based learning
        if not self.use_multiscale:
            # if we don't use multiscale, we only use the first layer
            ts_feat = ts_emb_1
            if self.use_prototype:
                ts_feat = rearrange(ts_feat, "b d tt -> b tt d")
                pe = self.pe(ts_feat)
                ts_pe = ts_feat + pe
                updates, attn = self.ts_grouping[0](ts_pe)
                slot_loss += torch.mean(attn)
                # last_ts_feat = torch.cat([updates, ts_feat], dim=1)
                ts_slot_list.append(updates)
                ts_feat_list.append(ts_feat)
            else:
                last_ts_feat = rearrange(ts_feat, "b d tt -> b tt d")
                ts_feat_list.append(last_ts_feat)
        else:
            # multi_scale_feats = []
            if self.use_prototype:
                for idx, ts_feat in enumerate([ts_emb_1, ts_emb_2, ts_emb_3]):
                    # extract the feature in each window
                    ts_feat = rearrange(ts_feat, "b d tt -> b tt d")
                    # position embedding
                    pe = self.pe(ts_feat)
                    ts_pe = ts_feat + pe
                    updates, attn = self.ts_grouping[idx](ts_pe)
                    slot_loss += torch.mean(attn)
                    # multi_scale_feats.append(updates)
                    # multi_scale_feats.append(ts_feat)
                    ts_slot_list.append(updates)
                    ts_feat_list.append(ts_feat)

                slot_loss /= len(ts_slot_list)
                # last_ts_feat = torch.cat(multi_scale_feats, dim=1)
            else:
                for idx, ts_feat in enumerate([ts_emb_1, ts_emb_2, ts_emb_3]):
                    # multi_scale_feats.append(
                    #     rearrange(ts_feat, "b d tt -> b tt d"))
                    ts_feat_list.append(rearrange(ts_feat, "b d tt -> b tt d"))

        # STEP 4: extract prototype features from images 
        # Only consider one scale for image.
        img_emb = self.img_conv_1(proj_x_img)
        img_feat = rearrange(img_emb, "b d tt -> b tt d")
        img_slot_list = []
        img_feat_list = []
        if self.use_prototype:
            pe = self.pe(img_feat)
            img_pe = img_feat + pe
            updates, attn = self.img_grouping(img_pe)
            slot_loss += torch.mean(attn)
            # last_img_feat = torch.cat([updates, img_feat], dim=1)
            img_slot_list.append(updates)
            img_feat_list.append(img_feat)
        else:
            img_feat_list.append(img_feat)
        
        # STEP 5: contrastive alignment 
        cont_loss = 0.
        avg_img_emb = rearrange(img_feat, "b tt d -> (b tt) d")
        for i, ts_feat in enumerate(ts_feat_list):
            ts_feat = rearrange(ts_feat.clone(), "b tt d -> b d tt")
            order = 2 ** i
            avg_ts_emb = F.avg_pool1d(ts_feat, kernel_size=(self.tt_max // (self.num_imgs * order)), 
                                      stride=(self.tt_max // (self.num_imgs * order)))
            avg_ts_emb = rearrange(avg_ts_emb, "b d tt -> (b tt) d")
            cont_loss += self.infonce_loss(avg_ts_emb, avg_img_emb)

        # STEP 6: fusion
        if self.use_prototype:
            concat_ts_slot = torch.cat(ts_slot_list, dim=1)
            concat_ts_feat = torch.cat(ts_feat_list, dim=1)
            concat_img_slot = torch.cat(img_slot_list, dim=1)
            concat_img_feat = torch.cat(img_feat_list, dim=1)
            concat_feat = torch.cat([concat_ts_slot, concat_ts_feat,
                                    concat_img_slot, concat_img_feat], dim=1)
        else:
            concat_ts_feat = torch.cat(ts_feat_list, dim=1)
            concat_img_feat = torch.cat(img_feat_list, dim=1)
            concat_feat = torch.cat([concat_ts_feat, concat_img_feat], dim=1)
        fusion_feat = self.fusion_layer(concat_feat)

        # STEP 7: make prediction
        if self.use_prototype:
            num_ts_tokens = concat_ts_slot.size(1) + concat_ts_feat.size(1)
        else:
            num_ts_tokens = concat_ts_feat.size(1)
        # num_img_tokens = concat_img_slot.size(1) + concat_img_feat.size(1)
        ts_pred_tokens = fusion_feat[:, :num_ts_tokens, :]
        attn, ts_pred_tokens = self.ts_atten_pooling(ts_pred_tokens)
        last_ts_feat = torch.bmm(attn.permute(0, 2, 1), ts_pred_tokens).squeeze(dim=1)
        img_pred_tokens = fusion_feat[:, num_ts_tokens:, :]
        attn, img_pred_tokens = self.img_atten_pooling(img_pred_tokens)
        last_img_feat = torch.bmm(attn.permute(0, 2, 1), img_pred_tokens).squeeze(dim=1)
        last_hs = torch.cat([last_ts_feat, last_img_feat], dim=1)

        # # attention pooling
        # if self.pooling_type == "attention":
        #     attn, last_ts_feat = self.atten_pooling(last_ts_feat)
        #     last_hs = torch.bmm(attn.permute(0, 2, 1),
        #                         last_ts_feat).squeeze(dim=1)
        # elif self.pooling_type == "mean":
        #     last_hs = last_ts_feat.mean(dim=1)
        # elif self.pooling_type == "last":
        #     last_hs = last_ts_feat[:, -1, :]

        # MLP for the final prediction
        # last_hs_proj = self.proj2(
        #     F.dropout(F.relu(self.proj1(last_hs)), p=self.dropout, training=self.training))
        # last_hs_proj += last_hs
        output = self.out_layer(last_hs)

        loss_dict = {
            "slot_loss": slot_loss,
            "cont_loss": cont_loss 
        }
        if self.task == 'ihm':
            if labels != None:
                ce_loss = self.loss_fct1(output, labels)
                loss_dict["ce_loss"] = ce_loss
                loss_dict["total_loss"] = ce_loss + self.lamb1 * slot_loss + cont_loss * self.lamb2
                return loss_dict
            return F.softmax(output, dim=-1)[:, 1]

        elif self.task == 'pheno':
            if labels != None:
                labels = labels.float()
                ce_loss = self.loss_fct1(output, labels)
                loss_dict["ce_loss"] = ce_loss
                loss_dict["total_loss"] = ce_loss + self.lamb1 * slot_loss + cont_loss * self.lamb2
                return loss_dict
            return torch.sigmoid(output)
        
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
    from torch.utils.data import DataLoader
    from cmehr.paths import *
    # from cmehr.dataset.mimic4_pretraining_datamodule import MIMIC4DataModule
    from cmehr.dataset.mimic4_downstream_datamodule import MIMIC4DataModule

    datamodule = MIMIC4DataModule(
        file_path=str(DATA_PATH / "output_mimic4/TS_CXR/ihm"),
        period_length=48
    )
    batch = dict()
    for batch in datamodule.val_dataloader():
        break
    for k, v in batch.items():
        if isinstance(v, Tensor):
            print(f"{k}: ", v.shape)

    """
    ts: torch.Size([4, 157, 17])
    ts_mask:  torch.Size([4, 157, 17])
    ts_tt:  torch.Size([4, 157])
    reg_ts:  torch.Size([4, 48, 34])
    cxr_imgs:  torch.Size([4, 5, 3, 512, 512])
    cxr_time:  torch.Size([4, 5])
    cxr_time_mask:  torch.Size([4, 5])
    reg_imgs:  torch.Size([4, 5, 3, 512, 512])
    reg_imgs_mask:  torch.Size([4, 5])
    label: torch.Size([4])
    """
    model = POCMPModule(
        use_multiscale=True,
        use_prototype=True
    )
    loss = model(
        x_ts=batch["ts"],  # type ignore
        x_ts_mask=batch["ts_mask"],
        ts_tt_list=batch["ts_tt"],
        reg_ts=batch["reg_ts"],
        cxr_imgs=batch["cxr_imgs"],
        cxr_time=batch["cxr_time"],
        cxr_time_mask=batch["cxr_time_mask"],
        reg_imgs=batch["reg_imgs"],
        reg_imgs_mask=batch["reg_imgs_mask"],
        labels=batch["label"],
    )
    print(loss)

    # feat1 = torch.randn(12, 128)
    # feat2 = torch.randn(48, 128)
    # coattn = OT_Attn_assem(impl="pot-uot-l2", ot_reg=0.05, ot_tau=0.5)
    # A_coattn, dist = coattn(feat1, feat2)
    # A_coattn: optimal transport matrix feat1 -> feat2
