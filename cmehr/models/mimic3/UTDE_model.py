import copy
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score
import math
from argparse import ArgumentParser, Namespace
from typing import Any, Tuple, Dict
from lightning import LightningModule
import ipdb
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer
from transformers import AutoModel
from .UTDE_modules import multiTimeAttention, gateMLP, BertForRepresentation, \
    MAGGate, Outer, TransformerEncoder, TransformerCrossEncoder, TimeSeriesCnnModel
from .interp import S_Interp, Cross_Interp, hold_out, recon_loss


class TSMixed(nn.Module):
    def __init__(self,
                 task: str = "ihm",
                 num_labels: int = 2,
                 modeltype: str = "TS",
                 num_heads: int = 8,
                 layers: int = 3,
                 kernel_size: int = 1,
                 dropout: float = 0.1,
                 irregular_learn_emb_ts: bool = True,
                 irregular_learn_emb_text: bool = True,
                 Interp: bool = False,
                 reg_ts: bool = True,
                 TS_mixup: bool = True,
                 mixup_level: str = "batch",
                 TS_model: str = "Atten",
                 tt_max: int = 48,
                 embed_time: int = 64,
                 embed_dim: int = 128,
                 orig_d_ts: int = 17,
                 orig_reg_d_ts: int = 34,
                 ts_seq_num: int = 48):

        super(TSMixed, self).__init__()

        self.modeltype = modeltype
        self.num_heads = num_heads
        self.attn_mask = False
        self.layers = layers
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.irregular_learn_emb_ts = irregular_learn_emb_ts
        self.irregular_learn_emb_text = irregular_learn_emb_text
        self.Interp = Interp
        self.reg_ts = reg_ts
        self.TS_mixup = TS_mixup
        self.mixup_level = mixup_level
        self.task = task
        self.TS_model = TS_model
        self.tt_max = tt_max

        self.time_query = torch.linspace(0, 1., self.tt_max)
        self.periodic = nn.Linear(1, embed_time-1)
        self.linear = nn.Linear(1, 1)
        output_dim = num_labels

        self.orig_d_ts = orig_d_ts
        self.d_ts = embed_dim
        self.ts_seq_num = ts_seq_num

        if self.Interp:
            self.s_intp = S_Interp(self.orig_d_ts, self.tt_max, embed_dim)
            self.c_intp = Cross_Interp(self.orig_d_ts)
            self.proj_ts_intp = nn.Conv1d(self.orig_d_ts*3, self.d_ts, kernel_size=self.kernel_size,
                                          padding=math.floor((self.kernel_size - 1) / 2), bias=False)

        if self.irregular_learn_emb_ts:
            self.time_attn_ts = multiTimeAttention(
                self.orig_d_ts*2,  self.d_ts, embed_time, 8)

        if self.reg_ts:
            self.orig_reg_d_ts = orig_reg_d_ts
            self.proj_ts = nn.Conv1d(self.orig_reg_d_ts, self.d_ts, kernel_size=self.kernel_size, padding=math.floor(
                (self.kernel_size - 1) / 2), bias=False)

        if self.TS_mixup:
            if self.mixup_level == 'batch':
                self.moe = gateMLP(
                    input_dim=self.d_ts*2, hidden_size=embed_dim, output_dim=1, dropout=self.dropout)
            elif self.mixup_level == 'batch_seq':
                self.moe = gateMLP(
                    input_dim=self.d_ts*2, hidden_size=embed_dim, output_dim=1, dropout=self.dropout)
            elif self.mixup_level == 'batch_seq_feature':
                self.moe = gateMLP(input_dim=self.d_ts*2, hidden_size=embed_dim,
                                   output_dim=self.d_ts, dropout=self.dropout)
            else:
                raise ValueError("Unknown mixedup type")

                # self.moe = nn.Linear(self.d_ts*self.tt_max*2, 1)
        if self.TS_model == 'LSTM':
            self.trans_ts_mem = nn.LSTM(input_size=self.d_ts, hidden_size=self.d_ts,
                                        num_layers=layers, dropout=self.dropout, bidirectional=True)

        elif self.TS_model == 'CNN':
            self.trans_ts_mem = TimeSeriesCnnModel(input_size=self.d_ts, n_filters=self.d_ts, filter_size=self.kernel_size,
                                                   dropout=self.dropout, length=self.tt_max, n_neurons=self.d_ts, layers=layers)
        elif self.TS_model == 'Atten':
            self.trans_ts_mem = self.get_network(
                self_type='ts_mem', layers=layers)

        self.proj1 = nn.Linear(self.d_ts, self.d_ts)
        self.proj2 = nn.Linear(self.d_ts, self.d_ts)
        self.out_layer = nn.Linear(self.d_ts, output_dim)

        if self.task == 'ihm':
            self.loss_fct1 = nn.CrossEntropyLoss()
        elif self.task == 'pheno':
            self.loss_fct1 = nn.BCEWithLogitsLoss()
        else:
            raise ValueError("Unknown task")

    def get_network(self, self_type='ts_mem', layers=-1):
        embed_dim = self.d_ts
        if self_type == 'ts_mem':
            if self.irregular_learn_emb_ts:
                q_seq_len = self.tt_max
            else:
                q_seq_len = self.ts_seq_num

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=layers,
                                  attn_dropout=self.dropout,
                                  relu_dropout=self.dropout,
                                  res_dropout=self.dropout,
                                  embed_dropout=self.dropout,
                                  attn_mask=self.attn_mask,
                                  q_seq_len=q_seq_len,
                                  kv_seq_len=None)

    def learn_time_embedding(self, tt):
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)

    def forward(self, x_ts, x_ts_mask, ts_tt_list, labels=None, reg_ts=None):
        """
        dimension [batch_size, seq_len, n_features]

        """
        if "TS" in self.modeltype:
            if self.Interp:
                x_ts_mask_interp = copy.deepcopy(x_ts_mask)
                x_ts_interp = copy.deepcopy(x_ts)
                recon_m = hold_out(x_ts_mask_interp)
                recon_m = torch.Tensor(recon_m).type_as(x_ts)
                proj_x_ts_interp = self.proj_ts_intp(self.c_intp(self.s_intp(
                    x_ts_interp, x_ts_mask_interp, ts_tt_list, recon_m)))  # dimension [batch_size,  n_features,seq_len]
                proj_x_ts_interp = proj_x_ts_interp.permute(2, 0, 1)
                recon_interp = self.c_intp(self.s_intp(
                    x_ts_interp, x_ts_mask_interp, ts_tt_list, recon_m, reconstruction=True), reconstruction=True)

            if self.irregular_learn_emb_ts:
                time_key_ts = self.learn_time_embedding(
                    ts_tt_list.type_as(x_ts))
                time_query = self.learn_time_embedding(
                    self.time_query.unsqueeze(0).type_as(x_ts))

                x_ts_irg = torch.cat((x_ts, x_ts_mask), 2)
                x_ts_mask = torch.cat((x_ts_mask, x_ts_mask), 2)

                proj_x_ts_irg = self.time_attn_ts(
                    time_query, time_key_ts, x_ts_irg, x_ts_mask)
                proj_x_ts_irg = proj_x_ts_irg.transpose(0, 1)

            if self.reg_ts and reg_ts != None:
                x_ts_reg = reg_ts.transpose(1, 2)
                proj_x_ts_reg = x_ts_reg if self.orig_reg_d_ts == self.d_ts else self.proj_ts(
                    x_ts_reg)
                proj_x_ts_reg = proj_x_ts_reg.permute(2, 0, 1)

            if self.TS_mixup:
                if self.Interp and not self.irregular_learn_emb_ts and self.reg_ts:
                    proj_x_ts_irg = proj_x_ts_interp
                if self.Interp and self.irregular_learn_emb_ts and not self.reg_ts:
                    proj_x_ts_reg = proj_x_ts_interp
                if self.mixup_level == 'batch':
                    g_irg = torch.max(proj_x_ts_irg, dim=0).values
                    g_reg = torch.max(proj_x_ts_reg, dim=0).values
                    moe_gate = torch.cat([g_irg, g_reg], dim=-1)
                elif self.mixup_level == 'batch_seq' or self.mixup_level == 'batch_seq_feature':
                    moe_gate = torch.cat(
                        [proj_x_ts_irg, proj_x_ts_reg], dim=-1)
                else:
                    raise ValueError("Unknown mixedup type")

                # for name, parameter in self.moe.named_parameters():
                mixup_rate = self.moe(moe_gate)
                proj_x_ts = mixup_rate*proj_x_ts_irg + \
                    (1-mixup_rate)*proj_x_ts_reg

            else:
                if self.irregular_learn_emb_ts:
                    proj_x_ts = proj_x_ts_irg
                elif self.reg_ts:
                    proj_x_ts = proj_x_ts_reg
                else:
                    raise ValueError("Unknown time series type")

            if self.TS_model == 'CNN':
                proj_x_ts = proj_x_ts.permute(1, 2, 0)
                proj_x_ts = self.trans_ts_mem(proj_x_ts)

            elif self.TS_model == 'LSTM':
                _, (proj_x_ts, _) = self.trans_ts_mem(proj_x_ts)
            else:
                proj_x_ts = self.trans_ts_mem(proj_x_ts)
            if self.TS_model != 'CNN':
                last_h_ts = proj_x_ts[-1]
            else:
                last_h_ts = proj_x_ts

            if self.modeltype == "TS":
                last_hs = last_h_ts
            else:
                raise ValueError("Unknown model type")

            last_hs_proj = self.proj2(F.dropout(
                F.relu(self.proj1(last_h_ts)), p=self.dropout, training=self.training))
            last_hs_proj += last_hs
            output = self.out_layer(last_hs_proj)

        if self.Interp:
            reconloss_interp = recon_loss(
                x_ts_interp, x_ts_mask_interp, recon_m, recon_interp, self.d_ts)

        if self.task == 'ihm':
            if labels != None:
                if self.Interp:
                    return self.loss_fct1(output, labels) + reconloss_interp
                else:
                    return self.loss_fct1(output, labels)
            return F.softmax(output, dim=-1)[:, 1]

        elif self.task == 'pheno':
            if labels != None:
                labels = labels.float()
                if self.Interp:
                    return self.loss_fct1(output, labels)+reconloss_interp
                else:
                    return self.loss_fct1(output, labels)
            return torch.sigmoid(output)


class MULTCrossModel(nn.Module):
    def __init__(self,
                 task: str = "ihm",
                 orig_d_ts: int = 17,
                 orig_reg_d_ts: int = 34,
                 orig_d_txt: int = 768,
                 tt_max: int = 48,
                 num_of_notes: int = 5,
                 modeltype: str = "TS_Text",
                 num_heads: int = 8,
                 layers: int = 3,
                 kernel_size: int = 1,
                 dropout: float = 0.1,
                 irregular_learn_emb_ts: bool = True,
                 irregular_learn_emb_text: bool = True,
                 reg_ts: bool = True,
                 TS_mixup: bool = True,
                 mixup_level: str = "batch",
                 cross_method: str = "self_cross",
                 embed_time: int = 64,
                 embed_dim: int = 128,
                 model_name: str = "yikuan8/Clinical-Longformer",
                 num_labels: int = 2,
                 cross_layers: int = 3,
                 ):
        """
        Construct a MulT Cross model.
        :task: str, the task of the model, e.g. "ihm", "pheno", "decomp", "los"
        :orig_d_ts: int, the original dimension of the time series data
        :orig_reg_d_ts: int, the original dimension of the regular time series data
        """
        super(MULTCrossModel, self).__init__()

        ts_seq_num = tt_max
        text_seq_num = num_of_notes

        self.modeltype = modeltype
        self.num_heads = num_heads
        self.layers = layers
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.attn_mask = False
        self.irregular_learn_emb_ts = irregular_learn_emb_ts
        self.irregular_learn_emb_text = irregular_learn_emb_text
        self.reg_ts = reg_ts
        self.TS_mixup = TS_mixup
        self.mixup_level = mixup_level
        self.task = task
        self.tt_max = tt_max
        self.cross_method = cross_method

        if self.irregular_learn_emb_ts or self.irregular_learn_emb_text:
            # formulate the regular time stamps
            self.time_query = torch.linspace(0, 1., self.tt_max)
            self.periodic = nn.Linear(1, embed_time-1)
            self.linear = nn.Linear(1, 1)

        if "TS" in self.modeltype:
            self.orig_d_ts = orig_d_ts
            self.d_ts = embed_dim
            self.ts_seq_num = ts_seq_num

            if self.irregular_learn_emb_ts:
                self.time_attn_ts = multiTimeAttention(
                    self.orig_d_ts*2, self.d_ts, embed_time, 8)

            if self.reg_ts:
                self.orig_reg_d_ts = orig_reg_d_ts
                self.proj_ts = nn.Conv1d(self.orig_reg_d_ts, self.d_ts, kernel_size=self.kernel_size, padding=math.floor(
                    (self.kernel_size - 1) / 2), bias=False)

            if self.TS_mixup:
                if self.mixup_level == 'batch':
                    self.moe = gateMLP(
                        input_dim=self.d_ts*2, hidden_size=embed_dim, output_dim=1, dropout=dropout)
                elif self.mixup_level == 'batch_seq':
                    self.moe = gateMLP(
                        input_dim=self.d_ts*2, hidden_size=embed_dim, output_dim=1, dropout=dropout)
                elif self.mixup_level == 'batch_seq_feature':
                    self.moe = gateMLP(
                        input_dim=self.d_ts*2, hidden_size=embed_dim, output_dim=self.d_ts, dropout=dropout)
                else:
                    raise ValueError("Unknown mixedup type")

        if "Text" in self.modeltype:
            Biobert = AutoModel.from_pretrained(model_name)
            self.orig_d_txt = orig_d_txt
            self.d_txt = embed_dim
            self.text_seq_num = text_seq_num
            self.bertrep = BertForRepresentation(model_name, Biobert)

            if self.irregular_learn_emb_text:
                self.time_attn = multiTimeAttention(
                    768, self.d_txt, embed_time, 8)
            else:
                self.proj_txt = nn.Conv1d(self.orig_d_txt, self.d_txt, kernel_size=self.kernel_size, padding=math.floor(
                    (self.kernel_size - 1) / 2), bias=False)

        output_dim = num_labels
        if self.modeltype == "TS_Text":
            if self.cross_method == "self_cross":
                self.trans_self_cross_ts_txt = self.get_cross_network(
                    layers=cross_layers)
                self.proj1 = nn.Linear(
                    self.d_ts+self.d_txt, self.d_ts+self.d_txt)
                self.proj2 = nn.Linear(
                    self.d_ts+self.d_txt, self.d_ts+self.d_txt)
                self.out_layer = nn.Linear(self.d_ts+self.d_txt, output_dim)
            else:
                self.trans_ts_mem = self.get_network(
                    self_type='ts_mem', layers=layers)
                self.trans_txt_mem = self.get_network(
                    self_type='txt_mem', layers=layers)

                if self.cross_method == "MulT":
                    self.trans_txt_with_ts = self.get_network(
                        self_type='txt_with_ts', layers=cross_layers)
                    self.trans_ts_with_txt = self.get_network(
                        self_type='ts_with_txt', layers=cross_layers)
                    self.proj1 = nn.Linear(
                        (self.d_ts+self.d_txt), (self.d_ts+self.d_txt))
                    self.proj2 = nn.Linear(
                        (self.d_ts+self.d_txt), (self.d_ts+self.d_txt))
                    self.out_layer = nn.Linear(
                        (self.d_ts+self.d_txt), output_dim)
                elif self.cross_method == "MAGGate":
                    self.gate_fusion = MAGGate(
                        inp1_size=self.d_txt, inp2_size=self.d_ts, dropout=self.embed_dropout)
                    self.proj1 = nn.Linear(self.d_txt, self.d_txt)
                    self.proj2 = nn.Linear(self.d_txt, self.d_txt)
                    self.out_layer = nn.Linear(self.d_txt, output_dim)
                elif self.cross_method == "Outer":
                    self.outer_fusion = Outer(
                        inp1_size=self.d_txt, inp2_size=self.d_ts)
                    self.proj1 = nn.Linear(self.d_txt, self.d_txt)
                    self.proj2 = nn.Linear(self.d_txt, self.d_txt)
                    self.out_layer = nn.Linear(self.d_txt, output_dim)
                else:
                    self.proj1 = nn.Linear(
                        self.d_ts+self.d_txt, self.d_ts+self.d_txt)
                    self.proj2 = nn.Linear(
                        self.d_ts+self.d_txt, self.d_ts+self.d_txt)
                    self.out_layer = nn.Linear(
                        self.d_ts+self.d_txt, output_dim)
        elif self.modeltype == "TS":
            self.proj1 = nn.Linear(self.d_ts, self.d_ts)
            self.proj2 = nn.Linear(self.d_ts, self.d_ts)
            self.out_layer = nn.Linear(self.d_ts, output_dim)
        elif self.modeltype == "Text":
            self.proj1 = nn.Linear(self.d_txt, self.d_txt)
            self.proj2 = nn.Linear(self.d_txt, self.d_txt)
            self.out_layer = nn.Linear(self.d_txt, output_dim)
        else:
            raise NotImplementedError

        if self.task == 'ihm':
            self.loss_fct1 = nn.CrossEntropyLoss()
        elif self.task == 'pheno':
            self.loss_fct1 = nn.BCEWithLogitsLoss()
        else:
            raise ValueError("Unknown task")

    def get_network(self, self_type='ts_mem', layers=-1):
        if self_type == 'ts_mem':
            if self.irregular_learn_emb_ts:
                embed_dim, q_seq_len, kv_seq_len = self.d_ts, self.tt_max, None
            else:
                embed_dim, q_seq_len, kv_seq_len = self.d_ts,  self.ts_seq_num, None
        elif self_type == 'txt_mem':
            if self.irregular_learn_emb_text:
                embed_dim, q_seq_len, kv_seq_len = self.d_txt, self.tt_max, None
            else:
                embed_dim, q_seq_len, kv_seq_len = self.d_txt, self.text_seq_num, None

        elif self_type == 'txt_with_ts':
            if self.irregular_learn_emb_ts:
                embed_dim,  q_seq_len, kv_seq_len = self.d_ts, self.tt_max, self.tt_max
            else:

                embed_dim, q_seq_len, kv_seq_len = self.d_ts, self.text_seq_num, self.ts_seq_num
        elif self_type == 'ts_with_txt':
            if self.irregular_learn_emb_text:
                embed_dim, q_seq_len, kv_seq_len = self.d_txt, self.tt_max, self.tt_max
            else:
                embed_dim, q_seq_len, kv_seq_len = self.d_txt, self.ts_seq_num, self.text_seq_num
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=layers,
                                  attn_dropout=self.dropout,
                                  relu_dropout=self.dropout,
                                  res_dropout=self.dropout,
                                  embed_dropout=self.dropout,
                                  attn_mask=self.attn_mask,
                                  q_seq_len=q_seq_len,
                                  kv_seq_len=kv_seq_len)

    def get_cross_network(self, layers=-1):
        embed_dim,  q_seq_len = self.d_ts, self.tt_max
        return TransformerCrossEncoder(embed_dim=embed_dim,
                                       num_heads=self.num_heads,
                                       layers=layers,
                                       attn_dropout=self.dropout,
                                       relu_dropout=self.dropout,
                                       res_dropout=self.dropout,
                                       embed_dropout=self.dropout,
                                       attn_mask=self.attn_mask,
                                       q_seq_len_1=q_seq_len)

    def learn_time_embedding(self, tt):
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)

    def forward(self, x_ts, x_ts_mask, ts_tt_list,
                input_ids_sequences, attn_mask_sequences,
                note_time_list, note_time_mask_list,
                labels=None, reg_ts=None):
        """ Forward function of Multimodal model
        :param x_ts: (B, N, D_t), torch.Tensor, time series data
        :param x_ts_mask: (B, N, D_t), torch.Tensor, time series mask
        :param ts_tt_list: (B, N), torch.Tensor, time series time
        :param input_ids_sequences: (B, N_text, L_text), torch.Tensor, text data
        :param attn_mask_sequences: (B, N_text, L_text), torch.Tensor, text mask
        :param note_time_list: (B, N_text), torch.Tensor, time of text data
        :param note_time_mask_list: (B, N_text), torch.Tensor, mask of text data
        :param labels: (B, ), torch.Tensor, labels
        :param reg_ts: (B, N_r, D_r), torch.Tensor, regular time series data
        """
        if "TS" in self.modeltype:
            if self.irregular_learn_emb_ts:
                # (B, N) -> (B, N, embed_time)
                time_key_ts = self.learn_time_embedding(
                    ts_tt_list)
                # (1, N_r) -> (1, N_r, embed_time)
                time_query = self.learn_time_embedding(
                    self.time_query.unsqueeze(0).type_as(x_ts))

                x_ts_irg = torch.cat((x_ts, x_ts_mask), 2)
                x_ts_mask = torch.cat((x_ts_mask, x_ts_mask), 2)

                # query: (1, N_r, embed_time),
                # key: (B, N, embed_time),
                # value: (B, N, 2 * D_t)
                # mask: (B, N, 2 * D_t)
                # out: (B, N_r, 128?)
                proj_x_ts_irg = self.time_attn_ts(
                    time_query, time_key_ts, x_ts_irg, x_ts_mask)
                proj_x_ts_irg = proj_x_ts_irg.transpose(0, 1)
            else:
                raise ValueError("Not implemented")

            if self.reg_ts and reg_ts != None:
                # convolution over regular time series
                x_ts_reg = reg_ts.transpose(1, 2)
                proj_x_ts_reg = x_ts_reg if self.orig_reg_d_ts == self.d_ts else self.proj_ts(
                    x_ts_reg)
                proj_x_ts_reg = proj_x_ts_reg.permute(2, 0, 1)
            else:
                raise ValueError("Not implemented")

            if self.TS_mixup:
                if self.mixup_level == 'batch':
                    g_irg = torch.max(proj_x_ts_irg, dim=0).values
                    g_reg = torch.max(proj_x_ts_reg, dim=0).values
                    moe_gate = torch.cat([g_irg, g_reg], dim=-1)
                elif self.mixup_level == 'batch_seq' or self.mixup_level == 'batch_seq_feature':
                    moe_gate = torch.cat(
                        [proj_x_ts_irg, proj_x_ts_reg], dim=-1)
                else:
                    raise ValueError("Unknown mixedup type")
                mixup_rate = self.moe(moe_gate)
                proj_x_ts = mixup_rate * proj_x_ts_irg + \
                    (1 - mixup_rate) * proj_x_ts_reg
            else:
                if self.irregular_learn_emb_ts:
                    proj_x_ts = proj_x_ts_irg
                elif self.reg_ts:
                    proj_x_ts = proj_x_ts_reg
                else:
                    raise ValueError("Unknown time series type")

        if "Text" in self.modeltype:
            # -> (B, N_text, D_text)
            x_txt = self.bertrep(input_ids_sequences, attn_mask_sequences)
            if self.irregular_learn_emb_text:
                # (B, N_text) -> (B, N_text, embed_time)
                time_key = self.learn_time_embedding(
                    note_time_list)
                if not self.irregular_learn_emb_ts:
                    time_query = self.learn_time_embedding(
                        self.time_query.unsqueeze(0))
                # (B, N_r, embed_time)
                proj_x_txt = self.time_attn(
                    time_query, time_key, x_txt, note_time_mask_list)
                proj_x_txt = proj_x_txt.transpose(0, 1)
            else:
                x_txt = x_txt.transpose(1, 2)
                proj_x_txt = x_txt if self.orig_d_txt == self.d_txt else self.proj_txt(
                    x_txt)
                proj_x_txt = proj_x_txt.permute(2, 0, 1)

        if self.modeltype == "TS_Text":
            if self.cross_method == "self_cross":
                hiddens = self.trans_self_cross_ts_txt([proj_x_txt, proj_x_ts])
                h_txt_with_ts, h_ts_with_txt = hiddens
                # (B, 2 * embed_time),
                # the global representation of each time series xxx
                last_hs = torch.cat(
                    [h_txt_with_ts[-1], h_ts_with_txt[-1]], dim=1)
            else:
                if self.cross_method == "MulT":
                    # ts --> txt
                    h_txt_with_ts = self.trans_txt_with_ts(
                        proj_x_txt, proj_x_ts, proj_x_ts)
                    # txt --> ts
                    h_ts_with_txt = self.trans_ts_with_txt(
                        proj_x_ts, proj_x_txt, proj_x_txt)
                    proj_x_ts = self.trans_ts_mem(h_txt_with_ts)
                    proj_x_txt = self.trans_txt_mem(h_ts_with_txt)

                    last_h_ts = proj_x_ts[-1]
                    last_h_txt = proj_x_txt[-1]
                    last_hs = torch.cat([last_h_ts, last_h_txt], dim=1)
                else:
                    proj_x_ts = self.trans_ts_mem(proj_x_ts)
                    proj_x_txt = self.trans_txt_mem(proj_x_txt)
                    if self.cross_method == "MAGGate":
                        last_hs = self.gate_fusion(
                            proj_x_txt[-1], proj_x_ts[-1])
                    elif self.cross_method == "Outer":
                        last_hs = self.outer_fusion(
                            proj_x_txt[-1], proj_x_ts[-1])
                    else:
                        last_hs = torch.cat(
                            [proj_x_txt[-1], proj_x_ts[-1]], dim=1)
        elif self.modeltype == "TS":
            last_hs = proj_x_ts.permute(1, 0, 2)
            # last_hs = torch.mean(last_hs, dim=1)
            # use the last timestamp?
            last_hs = last_hs[:, -1]
        elif self.modeltype == "Text":
            last_hs = proj_x_txt.permute(1, 0, 2)
            # last_hs = torch.mean(last_hs, dim=1)
            last_hs = last_hs[:, -1]
        else:
            raise NotImplementedError

        last_hs_proj = self.proj2(
            F.dropout(F.relu(self.proj1(last_hs)), p=self.dropout, training=self.training))
        last_hs_proj += last_hs
        output = self.out_layer(last_hs_proj)

        if torch.isnan(output).any():
            ipdb.set_trace()

        if self.task == 'ihm':
            if labels != None:
                return self.loss_fct1(output, labels)
            return F.softmax(output, dim=-1)[:, 1]

        elif self.task == 'pheno':
            if labels != None:
                labels = labels.float()
                return self.loss_fct1(output, labels)
            return torch.sigmoid(output)


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from cmehr.paths import *
    from cmehr.dataset.UTDE_datamodule import TSNote_Irg, TextTSIrgcollate_fn

    dataset = TSNote_Irg(
        file_path=str(ROOT_PATH / "output/ihm"),
        split="train",
        bert_type="yikuan8/Clinical-Longformer",
        max_length=128
    )

    dataloader = DataLoader(dataset=dataset,
                            batch_size=4,
                            num_workers=1,
                            shuffle=True,
                            collate_fn=TextTSIrgcollate_fn)
    assert len(dataloader) > 0
    batch = dict()
    for batch in dataloader:
        break

    for k, v in batch.items():  # type: ignore
        print(k, ": ", v.shape)

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
    model = MULTCrossModel()
    loss = model(
        x_ts=batch["ts"],  # type ignore
        x_ts_mask=batch["ts_mask"],
        ts_tt_list=batch["ts_tt"],
        reg_ts=batch["reg_ts"],
        input_ids_sequences=batch["input_ids"],
        attn_mask_sequences=batch["attention_mask"],
        note_time_list=batch["note_time"],
        note_time_mask_list=batch["note_time_mask"],
        labels=batch["label"],
    )
    print(loss)
