import numpy as np
from PIL import Image
import math
import ot
import ipdb
from einops import rearrange
from typing import Optional, List
from torch import nn
from transformers import AutoModel
import torch.nn.functional as F
import torch
from cmehr.models.mimic4.UTDE_modules import multiTimeAttention
from cmehr.models.mimic4.base_model import MIMIC4LightningModule
from cmehr.backbone.vision.pretrained import get_biovil_t_image_encoder


class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)  # 4, 50, 64
        A = self.attention_c(A)  # N x n_classes
        return A, x


class MIMIC4MTANDModule(MIMIC4LightningModule):
    def __init__(self,
                 task: str = "ihm",
                 modeltype: str = "CXR",
                 max_epochs: int = 10,
                 ts_learning_rate: float = 4e-4,
                 period_length: int = 48,
                 num_heads: int = 8,
                 layers: int = 3,
                 dropout: float = 0.1,
                 irregular_learn_emb_img: bool = True,
                 reg_ts: bool = True,
                 pooling_type: str = "mean",
                 lamb: float = 1.,
                 embed_time: int = 64,
                 embed_dim: int = 128,
                 *args,
                 **kwargs
                 ):
        super().__init__(task=task, modeltype="CXR", max_epochs=max_epochs,
                         ts_learning_rate=ts_learning_rate, period_length=period_length)
        self.save_hyperparameters()

        self.num_heads = num_heads
        self.layers = layers
        self.dropout = dropout
        self.irregular_learn_emb_img = irregular_learn_emb_img
        self.task = task
        self.tt_max = period_length
        self.embed_dim = embed_dim
        self.pooling_type = pooling_type
        self.lamb = lamb

        self.img_encoder = get_biovil_t_image_encoder()
        self.img_embed_dim = 512

        if self.irregular_learn_emb_img:
            # formulate the regular time stamps
            self.periodic = nn.Linear(1, embed_time-1)
            self.linear = nn.Linear(1, 1)
            self.time_query = torch.linspace(0, 1., self.tt_max)
            self.time_attn = multiTimeAttention(
                self.img_embed_dim, self.embed_dim, embed_time, 8)
            
        self.proj1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.proj2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_layer = nn.Linear(self.embed_dim, self.num_labels)

        self.atten_pooling = Attn_Net_Gated(
            L=embed_dim, D=64, dropout=True, n_classes=1)

    def learn_time_embedding(self, tt):
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)


    def forward(self,
                cxr_imgs: torch.Tensor,
                cxr_time: torch.Tensor,
                cxr_time_mask: torch.Tensor,
                labels=None,
                **kwargs):
        
        batch_size = cxr_imgs.size(0)
        flatten_imgs = rearrange(cxr_imgs, 'b n c h w -> (b n) c h w')
        x_img = self.img_encoder(flatten_imgs).img_embedding
        x_img = rearrange(x_img, '(b n) c -> b n c', b=batch_size)

        if self.irregular_learn_emb_img:
            time_query = self.learn_time_embedding(
                self.time_query.unsqueeze(0).type_as(x_img))
            # (B, N_text) -> (B, N_text, embed_time)
            time_key = self.learn_time_embedding(
                cxr_time)
            # (B, N_r, embed_time)
            proj_x_img = self.time_attn(
                time_query, time_key, x_img, cxr_time_mask)
            proj_x_img = proj_x_img.transpose(0, 1)
        else:
            x_img = x_img.transpose(1, 2)
            proj_x_img = x_img if self.orig_d_txt == self.d_txt else self.proj_txt(
                x_img)
            proj_x_img = proj_x_img.permute(2, 0, 1)

        last_feat = proj_x_img.permute(1, 0, 2)
        # attention pooling
        if self.pooling_type == "attention":
            attn, last_feat = self.atten_pooling(last_feat)
            last_hs = torch.bmm(attn.permute(0, 2, 1),
                                last_feat).squeeze(dim=1)
        elif self.pooling_type == "mean":
            last_hs = last_feat.mean(dim=1)
        elif self.pooling_type == "last":
            last_hs = last_feat[:, -1, :]

        # MLP for the final prediction
        last_hs_proj = self.proj2(
            F.dropout(F.relu(self.proj1(last_hs)), p=self.dropout, training=self.training))
        last_hs_proj += last_hs
        output = self.out_layer(last_hs_proj)

        if self.task in ['ihm', 'readm']:
            if labels != None:
                ce_loss = self.loss_fct1(output, labels)
                return ce_loss
            return F.softmax(output, dim=-1)[:, 1]

        elif self.task == 'pheno':
            if labels != None:
                labels = labels.float()
                ce_loss = self.loss_fct1(output, labels)
                return ce_loss
            return torch.sigmoid(output)


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from cmehr.paths import *
    # from cmehr.dataset.mimic4_pretraining_datamodule import MIMIC4DataModule
    # from cmehr.dataset.mimic3_downstream_datamodule import MIMIC3DataModule
    from cmehr.dataset.mimic4_downstream_datamodule import MIMIC4DataModule

    datamodule = MIMIC4DataModule(
        mimic_cxr_dir=str(MIMIC_CXR_JPG_PATH),
        file_path=str(DATA_PATH / "output_mimic4/TS_CXR/pheno"),
        period_length=48
    )
    batch = dict()
    for batch in datamodule.val_dataloader():
        break
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: ", v.shape)

    """
    ts:  torch.Size([4, 64, 15])
    ts_mask:  torch.Size([4, 64, 15])
    ts_tt:  torch.Size([4, 64])
    reg_ts:  torch.Size([4, 24, 30])
    cxr_imgs:  torch.Size([4, 5, 3, 512, 512])
    cxr_time:  torch.Size([4, 5])
    cxr_time_mask:  torch.Size([4, 5])
    reg_imgs:  torch.Size([4, 5, 3, 512, 512])
    reg_imgs_mask:  torch.Size([4, 5])
    label:  torch.Size([4, 25])
    """
    model = MIMIC4MTANDModule()
    loss = model(
        cxr_imgs=batch["cxr_imgs"],
        cxr_time=batch["cxr_time"],
        cxr_time_mask=batch["cxr_time_mask"],
    )
    print(loss)

    # feat1 = torch.randn(12, 128)
    # feat2 = torch.randn(48, 128)
    # coattn = OT_Attn_assem(impl="pot-uot-l2", ot_reg=0.05, ot_tau=0.5)
    # A_coattn, dist = coattn(feat1, feat2)
    # A_coattn: optimal transport matrix feat1 -> feat2
