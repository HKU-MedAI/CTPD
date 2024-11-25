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
# from cmehr.models.mimic4.base_model import MIMIC4LightningModule
from cmehr.models.mimic3.base_model import MIMIC3NoteModule
from cmehr.models.mimic4.UTDE_modules import BertForRepresentation


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


class MTANDModule(MIMIC3NoteModule):
    def __init__(self,
                 task: str = "ihm",
                 modeltype: str = "TS",
                 max_epochs: int = 10,
                 ts_learning_rate: float = 4e-4,
                 period_length: int = 48,
                 num_heads: int = 8,
                 layers: int = 3,
                 dropout: float = 0.1,
                 irregular_learn_emb_text: bool = True,
                 reg_ts: bool = True,
                 pooling_type: str = "mean",
                 lamb: float = 1.,
                 embed_time: int = 64,
                 embed_dim: int = 128,
                 bert_type: str = "prajjwal1/bert-tiny",
                 *args,
                 **kwargs
                 ):
        super().__init__(task=task, modeltype=modeltype, max_epochs=max_epochs,
                         ts_learning_rate=ts_learning_rate, period_length=period_length)
        self.save_hyperparameters()

        self.num_heads = num_heads
        self.layers = layers
        self.dropout = dropout
        self.irregular_learn_emb_text = irregular_learn_emb_text
        self.task = task
        self.tt_max = period_length
        self.d_txt = embed_dim
        self.pooling_type = pooling_type
        self.lamb = lamb

        if self.irregular_learn_emb_text:
            # formulate the regular time stamps
            self.periodic = nn.Linear(1, embed_time-1)
            self.linear = nn.Linear(1, 1)
            self.time_query = torch.linspace(0, 1., self.tt_max)
            if bert_type == "prajjwal1/bert-tiny":
                self.time_attn = multiTimeAttention(
                    128, self.d_txt, embed_time, 8)
            elif bert_type == "yikuan8/Clinical-Longformer":
                self.time_attn = multiTimeAttention(
                    512, self.d_txt, embed_time, 8)

        Biobert = AutoModel.from_pretrained(bert_type)
        self.bertrep = BertForRepresentation(bert_type, Biobert)

        self.proj1 = nn.Linear(self.d_txt, self.d_txt)
        self.proj2 = nn.Linear(self.d_txt, self.d_txt)
        self.out_layer = nn.Linear(self.d_txt, self.num_labels)

        self.atten_pooling = Attn_Net_Gated(
            L=embed_dim, D=64, dropout=True, n_classes=1)

    def learn_time_embedding(self, tt):
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)


    def forward(self,
                input_ids_sequences=None,
                attn_mask_sequences=None,
                note_time_list=None,
                note_time_mask_list=None,
                labels=None, reg_ts=None,
                **kwargs):
        
        x_txt = self.bertrep(input_ids_sequences, attn_mask_sequences)
        if self.irregular_learn_emb_text:
            time_query = self.learn_time_embedding(
                self.time_query.unsqueeze(0).type_as(x_txt))
            # (B, N_text) -> (B, N_text, embed_time)
            time_key = self.learn_time_embedding(
                note_time_list)
            # (B, N_r, embed_time)
            proj_x_txt = self.time_attn(
                time_query, time_key, x_txt, note_time_mask_list)
            proj_x_txt = proj_x_txt.transpose(0, 1)
        else:
            x_txt = x_txt.transpose(1, 2)
            proj_x_txt = x_txt if self.orig_d_txt == self.d_txt else self.proj_txt(
                x_txt)
            proj_x_txt = proj_x_txt.permute(2, 0, 1)

        last_feat = proj_x_txt.permute(1, 0, 2)
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

    def recon_prototype_embeds(self, ts_feat1: torch.Tensor, x_len: List):
        '''
        We are not using this function at this time.
        ts_feat_1: (B, D, TT)
        prototype_embeds: (B, D)
        '''
        pass
        # all_rec_feat = []
        # for i in range(batch_size):
        #     ts_atten_scale1, _ = self.ot_coattn(
        #         ts_feat_list[1][:, i], ts_feat_list[0][:, i])
        #     rec_feat_1 = ts_atten_scale1 @ ts_feat_list[0][:, i]
        #     rec_feat_1 = torch.cat([rec_feat_1, ts_feat_list[1][:, i]], dim=0)

        #     ts_atten_scale2, _ = self.ot_coattn(
        #         ts_feat_list[2][:, i], rec_feat_1)
        #     rec_feat_2 = ts_atten_scale2 @ rec_feat_1
        #     rec_feat_2 = torch.cat([rec_feat_2, ts_feat_list[2][:, i]], dim=0)
        #     all_rec_feat.append(rec_feat_2)

        # all_rec_feat = torch.stack(all_rec_feat, dim=0)
        # all_rec_feat = all_rec_feat.mean(dim=1)

        # return all_rec_feat, 0

        # flatten_feat1 = rearrange(ts_feat1, "b tt d -> (b tt) d")
        # ts_atten_scale1, loss_ot = self.ot_coattn(
        #     flatten_feat1, self.scale1_concepts)
        # rec_ts_feat1 = rearrange(
        #     ts_atten_scale1 @ self.scale1_concepts, "(b tt) d -> b tt d", b=batch_size)
        # rec_ts_emb1 = rec_ts_feat1.mean(dim=1)

        # rec_ts_feat1, score_feat1 = self.grouping(ts_feat1)
        # updates, atten1 = self.grouping(ts_feat1)
        # cpt_activation = attn
        # # attention pooling
        # atten_score, rec_ts_feat1 = self.atten_pooling(rec_ts_feat1)
        # ts_pool = torch.bmm(atten_score.permute(
        #     0, 2, 1), rec_ts_feat1).squeeze(1)

        # return ts_pool, 0


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from cmehr.paths import *
    # from cmehr.dataset.mimic4_pretraining_datamodule import MIMIC4DataModule
    from cmehr.dataset.mimic3_downstream_datamodule import MIMIC3DataModule

    datamodule = MIMIC3DataModule(
        file_path=str(DATA_PATH / "output_mimic3/pheno"),
        tt_max=48,
        bert_type="prajjwal1/bert-tiny",
        max_length=512
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
    model = MTANDModule(
    )
    loss = model(
        input_ids_sequences=batch["input_ids"],
        attn_mask_sequences=batch["attention_mask"],
        note_time_list=batch["note_time"],
        note_time_mask_list=batch["note_time_mask"],
    )
    print(loss)

    # feat1 = torch.randn(12, 128)
    # feat2 = torch.randn(48, 128)
    # coattn = OT_Attn_assem(impl="pot-uot-l2", ot_reg=0.05, ot_tau=0.5)
    # A_coattn, dist = coattn(feat1, feat2)
    # A_coattn: optimal transport matrix feat1 -> feat2
