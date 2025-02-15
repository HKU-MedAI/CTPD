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
from cmehr.models.mimic3.CTPD_model import SlotAttention


class CTPDModule(MIMIC4LightningModule):
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
                 num_slots: int = 20,
                 period_length: float = 48,
                 lamb1: float = 1.,
                 lamb2: float = 1.,
                 lamb3: float = 1.,
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
        self.lamb3 = lamb3
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
        self.time_query = torch.linspace(0, 1., self.tt_max)
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
                # self.moe_img = gateMLP(
                #     input_dim=self.embed_dim*2, hidden_size=embed_dim, output_dim=1, dropout=dropout)
            # elif self.mixup_level == 'batch_seq':
            #     self.moe = gateMLP(
            #         input_dim=self.embed_dim*2, hidden_size=embed_dim, output_dim=1, dropout=dropout)
            # elif self.mixup_level == 'batch_seq_feature':
            #     self.moe = gateMLP(
            #         input_dim=self.embed_dim*2, hidden_size=embed_dim, output_dim=self.embed_dim, dropout=dropout)
            else:
                raise ValueError("Unknown mixedup type")

        self.proj_reg_ts = nn.Conv1d(orig_reg_d_ts, self.embed_dim, kernel_size=1, padding=0, bias=False)
        # self.proj_reg_img = nn.Conv1d(self.embed_dim, self.embed_dim, kernel_size=1, padding=0, bias=False)

        if self.use_prototype:
            self.pe = PositionalEncoding1D(embed_dim)

            # define shared slots
            self.num_slots = num_slots
            self.shared_slots_mu = nn.Parameter(torch.randn(1, 1, self.embed_dim))
            self.shared_slots_logsigma = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            nn.init.xavier_uniform_(self.shared_slots_logsigma)

            # if self.use_multiscale:
            #     num_prototypes = [20, 10, 5]
            # else:
            #     num_prototypes = [10]
            # self.ts_grouping = []
            # for i, num_prototype in enumerate(num_prototypes):
            #     self.ts_grouping.append(SlotAttention(
            #         dim=embed_dim, num_slots=num_prototype))
            # self.ts_grouping = nn.ModuleList(self.ts_grouping)
            # self.img_grouping = SlotAttention(
            #     dim=embed_dim, num_slots=10)

            # define slot attention modules to discover correspondence
            self.ts_grouping = SlotAttention(dim=embed_dim)
            self.img_grouping = SlotAttention(dim=embed_dim)

            # gating mechanism
            self.weight_proj = nn.Sequential(
                nn.Linear(int(self.embed_dim * 2), self.num_slots),
                nn.ReLU(inplace=True),
                nn.Linear(self.num_slots, self.num_slots)
            )

        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4)
        self.fusion_layer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.ts_atten_pooling = Attn_Net_Gated(
            L=embed_dim, D=64, dropout=True, n_classes=1)
        self.img_atten_pooling = Attn_Net_Gated(
            L=embed_dim, D=64, dropout=True, n_classes=1)

        decoder_layer = nn.TransformerDecoderLayer(d_model=self.embed_dim, nhead=8, batch_first=True)
        self.ts_decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        self.img_decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        self.ts_proj = nn.Linear(self.embed_dim, self.orig_reg_d_ts)

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
            self.time_query.unsqueeze(0).type_as(x_ts))
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
        proj_x_ts_reg = self.proj_reg_ts(x_ts_reg)
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
            self.time_query.unsqueeze(0).type_as(x_img))
        mask_img = cxr_time_mask.unsqueeze(-1).expand(-1, -1, self.embed_dim)
        proj_x_img_irg = self.time_attn_img(
            time_query, time_key_img, x_img, mask_img)
        proj_x_img_irg = proj_x_img_irg.transpose(0, 1)

        return proj_x_img_irg

    # def forward_ts_reg(self, reg_ts: torch.Tensor):
    #     '''
    #     Forward regular time series.
    #     '''
    #     # convolution over regular time series
    #     x_ts_reg = reg_ts.transpose(1, 2)
    #     proj_x_ts_reg = self.proj_reg_ts(x_ts_reg)
    #     proj_x_ts_reg = proj_x_ts_reg.permute(2, 0, 1)

    #     return proj_x_ts_reg
    
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

    def infonce_loss(self, sim, temperature=0.07):
        """
        Compute the InfoNCE loss for the given outputs.
        """
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
        ts_feat_list = []
        # STEP 3: prototype-based learning
        ts_feat_list = []
        if not self.use_multiscale:
            # if we don't use multiscale, we only use the first layer
            ts_feat = ts_emb_1
            ts_feat = rearrange(ts_feat, "b d tt -> b tt d")
            ts_feat_list.append(ts_feat)
        else:
            for idx, ts_feat in enumerate([ts_emb_1, ts_emb_2, ts_emb_3]):
                # extract the feature in each window
                ts_feat = rearrange(ts_feat, "b d tt -> b tt d")
                ts_feat_list.append(ts_feat)

        # STEP 4: extract prototype features from images 
        # Only consider one scale for image.
        img_emb = self.img_conv_1(proj_x_img)
        img_feat = rearrange(img_emb, "b d tt -> b tt d")
        img_feat_list = []
        img_feat_list.append(img_feat)
        
        ts_feat_concat = torch.cat(ts_feat_list, dim=1)
        img_feat_concat = torch.cat(img_feat_list, dim=1)
        
        cont_loss = torch.tensor(0.).type_as(x_ts)
        # recon_loss = torch.tensor(0.).type_as(x_ts)
        ts_recon_loss = torch.tensor(0.).type_as(x_ts)
        img_recon_loss = torch.tensor(0.).type_as(x_ts)
        if self.use_prototype:
            batch_size = x_ts.size(0)
            shared_mu = self.shared_slots_mu.expand(batch_size, self.num_slots, -1)
            shared_sigma = self.shared_slots_logsigma.exp().expand(batch_size, self.num_slots, -1)
            shared_slots = shared_mu + shared_sigma * torch.randn(shared_mu.shape).type_as(x_ts)
            # ts_mu = self.ts_slots_mu.expand(batch_size, self.num_slots, -1)
            # ts_sigma = self.ts_slots_logsigma.exp().expand(batch_size, self.num_slots, -1)
            # ts_slots = ts_mu + ts_sigma * torch.randn(ts_mu.shape).type_as(x_ts)
            # img_mu = self.img_slots_mu.expand(batch_size, self.num_slots, -1)
            # img_sigma = self.img_slots_logsigma.exp().expand(batch_size, self.num_slots, -1)
            # img_slots = img_mu + img_sigma * torch.randn(img_mu.shape).type_as(x_ts)

            pe = self.pe(ts_feat_concat)
            ts_pe = ts_feat_concat + pe
            shared_ts_slots, _ = self.ts_grouping(shared_slots, ts_pe)
            shared_ts_slots = F.normalize(shared_ts_slots, dim=-1)
            # ts_slots, _ = self.ts_grouping(ts_slots, ts_pe)
            # ts_slots = F.normalize(ts_slots, dim=-1)
            
            pe = self.pe(img_feat_concat)
            img_pe = img_feat_concat + pe
            shared_img_slots, _ = self.img_grouping(shared_slots, img_pe)
            shared_img_slots = F.normalize(shared_img_slots, dim=-1)
            # img_slots, _ = self.img_grouping(img_slots, img_pe)
            # img_slots = F.normalize(img_slots, dim=-1)

            # compute contrastive loss
            global_ts_feat = shared_ts_slots.mean(dim=1)
            global_img_feat = shared_img_slots.mean(dim=1)
            global_feat = torch.cat([global_ts_feat, global_img_feat], dim=1)
            slot_weights = F.softmax(self.weight_proj(global_feat), dim=-1)
            pair_similarity = torch.einsum('bnd,cnd->bcn', shared_ts_slots, shared_img_slots)
            similarity = torch.einsum('bcn,bn->bc', pair_similarity, slot_weights)
            cont_loss = self.infonce_loss(similarity, temperature=0.2)

            # concat_ts_slot = torch.cat([shared_ts_slots, ts_slots], dim=1)
            concat_ts_slot = shared_ts_slots
            concat_ts_feat = torch.cat(ts_feat_list, dim=1)
            # concat_img_slot = torch.cat([shared_img_slots, img_slots], dim=1)
            concat_img_slot = shared_img_slots
            concat_img_feat = torch.cat(img_feat_list, dim=1)
            # both slots and timestamp-level embeddings are used
            concat_feat = torch.cat([concat_ts_slot, concat_ts_feat,
                                     concat_img_slot, concat_img_feat], dim=1)
            # concat_feat = torch.cat([concat_ts_slot, concat_img_slot], dim=1)

            # using predicted slot features to reconstruct regular time series
            ts_tgt_embs = rearrange(proj_x_ts_reg, "tt b d -> b tt d")
            mask = nn.Transformer.generate_square_subsequent_mask(self.tt_max - 1)
            pred_ts_emb = self.ts_decoder(tgt=ts_tgt_embs[:, :-1], memory=concat_ts_slot, tgt_mask=mask, tgt_is_causal=True)
            pred_ts = self.ts_proj(pred_ts_emb)
            ts_recon_loss = F.mse_loss(pred_ts, reg_ts[:, 1:])

            img_tgt_embs = rearrange(proj_x_img_irg, "tt b d -> b tt d")
            pred_img_emb = self.img_decoder(tgt=img_tgt_embs[:, :-1], memory=concat_img_slot, tgt_mask=mask, tgt_is_causal=True)
            img_recon_loss = F.mse_loss(pred_img_emb, img_tgt_embs[:, 1:])
        else:
            concat_ts_feat = torch.cat(ts_feat_list, dim=1)
            concat_img_feat = torch.cat(img_feat_list, dim=1)
            concat_feat = torch.cat([concat_ts_feat, concat_img_feat], dim=1)

        fusion_feat = self.fusion_layer(concat_feat)

        # STEP 7: make prediction
        if self.use_prototype:
            num_ts_tokens = concat_ts_slot.size(1) + concat_ts_feat.size(1)
            # num_ts_tokens = concat_ts_slot.size(1)
        else:
            num_ts_tokens = concat_ts_feat.size(1)
        
        # num_text_tokens = concat_text_slot.size(1) + concat_text_feat.size(1)
        last_ts_feat = torch.mean(fusion_feat[:, :num_ts_tokens, :], dim=1)
        last_img_feat = torch.mean(fusion_feat[:, num_ts_tokens:, :], dim=1)
        last_hs = torch.cat([last_ts_feat, last_img_feat], dim=1)

        output = self.out_layer(last_hs)
        loss_dict = {
            "cont_loss": cont_loss,
            "ts_recon_loss": ts_recon_loss,
            "text_recon_loss": img_recon_loss,
        }
        if self.task in ['ihm', 'readm']:
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
    model = CTPDModule(
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
