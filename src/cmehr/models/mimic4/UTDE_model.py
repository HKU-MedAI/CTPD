from einops import rearrange
import numpy as np
import math
import ipdb
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer
from cmehr.models.mimic4.UTDE_modules import multiTimeAttention, gateMLP, \
    MAGGate, Outer, TransformerEncoder, TransformerCrossEncoder
from cmehr.models.mimic4.base_model import MIMIC4LightningModule
# from cmehr.backbone import get_biovil_t_image_encoder


class UTDEModule(MIMIC4LightningModule):
    def __init__(self,
                 task: str = "ihm",
                 orig_d_ts: int = 17,
                 orig_reg_d_ts: int = 34,
                 num_heads: int = 8,
                 layers: int = 3,
                 kernel_size: int = 1,
                 dropout: float = 0.1,
                 irregular_learn_emb_ts: bool = True,
                 irregular_learn_emb_img: bool = True,
                 reg_ts: bool = True,
                 TS_mixup: bool = True,
                 mixup_level: str = "batch",
                 cross_method: str = "self_cross",
                 embed_time: int = 64,
                 embed_dim: int = 128,
                 num_labels: int = 2,
                 cross_layers: int = 3,
                 max_epochs: int = 10,
                 img_learning_rate: float = 1e-4,
                 ts_learning_rate: float = 4e-4,
                 period_length: int = 48,
                 *args,
                 **kwargs
                 ):
        
        super().__init__(task=task, max_epochs=max_epochs,
                         modeltype="TS",
                         img_learning_rate=img_learning_rate,
                         ts_learning_rate=ts_learning_rate,
                         period_length=period_length,
                         num_labels=num_labels,
                         )

        ts_seq_num = period_length
        self.num_heads = num_heads
        self.layers = layers
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.attn_mask = False
        self.irregular_learn_emb_ts = irregular_learn_emb_ts
        self.irregular_learn_emb_img = irregular_learn_emb_img
        self.reg_ts = reg_ts
        self.TS_mixup = TS_mixup
        self.mixup_level = mixup_level
        self.task = task
        self.tt_max = period_length
        self.cross_method = cross_method

        if self.irregular_learn_emb_ts or self.irregular_learn_emb_img:
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
    
        self.proj1 = nn.Linear(self.d_ts, self.d_ts)
        self.proj2 = nn.Linear(self.d_ts, self.d_ts)
        self.out_layer = nn.Linear(self.d_ts, self.num_labels)

        # if self.task in ['ihm', 'readm']:
        #     self.loss_fct1 = nn.CrossEntropyLoss()
        # elif self.task == 'pheno':
        #     self.loss_fct1 = nn.BCEWithLogitsLoss()
        # else:
        #     raise ValueError("Unknown task")

    def get_network(self, self_type='ts_mem', layers=-1):
        if self_type == 'ts_mem':
            if self.irregular_learn_emb_ts:
                embed_dim, q_seq_len, kv_seq_len = self.d_ts, self.tt_max, None
            else:
                embed_dim, q_seq_len, kv_seq_len = self.d_ts,  self.ts_seq_num, None
        # elif self_type == 'txt_mem':
        #     if self.irregular_learn_emb_img:
        #         embed_dim, q_seq_len, kv_seq_len = self.d_img, self.tt_max, None
        #     else:
        #         embed_dim, q_seq_len, kv_seq_len = self.d_img, self.img_seq_num, None

        # elif self_type == 'txt_with_ts':
        #     if self.irregular_learn_emb_ts:
        #         embed_dim,  q_seq_len, kv_seq_len = self.d_ts, self.tt_max, self.tt_max
        #     else:

        #         embed_dim, q_seq_len, kv_seq_len = self.d_ts, self.img_seq_num, self.ts_seq_num
        # elif self_type == 'ts_with_txt':
        #     if self.irregular_learn_emb_img:
        #         embed_dim, q_seq_len, kv_seq_len = self.d_img, self.tt_max, self.tt_max
        #     else:
        #         embed_dim, q_seq_len, kv_seq_len = self.d_img, self.ts_seq_num, self.img_seq_num
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
                labels=None, reg_ts=None):
        """ Forward function of Multimodal model
        :param x_ts: (B, N, D_t), torch.Tensor, time series data
        :param x_ts_mask: (B, N, D_t), torch.Tensor, time series mask
        :param ts_tt_list: (B, N), torch.Tensor, time series time
        :param labels: (B, ), torch.Tensor, labels
        :param reg_ts: (B, N_r, D_r), torch.Tensor, regular time series data
        """

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

        last_hs = proj_x_ts.permute(1, 0, 2)
        # last_hs = torch.mean(last_hs, dim=1)
        # use the last timestamp?
        last_hs = last_hs[:, -1]

        last_hs_proj = self.proj2(
            F.dropout(F.relu(self.proj1(last_hs)), p=self.dropout, training=self.training))
        last_hs_proj += last_hs
        output = self.out_layer(last_hs_proj)

        if torch.isnan(output).any():
            ipdb.set_trace()

        if self.task in ['ihm', 'readm']:
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
    # from cmehr.dataset.UTDE_datamodule import TSNote_Irg, TextTSIrgcollate_fn
    # from cmehr.dataset import MIMIC4DataModule
    from cmehr.dataset.mimic3_downstream_datamodule import MIMIC3DataModule

    # dataset = TSNote_Irg(
    #     file_path=str(ROOT_PATH / "output/ihm"),
    #     split="train",
    #     bert_type="yikuan8/Clinical-Longformer",
    #     max_length=128
    # )

    # dataloader = DataLoader(dataset=dataset,
    #                         batch_size=4,
    #                         num_workers=1,
    #                         shuffle=True,
    #                         collate_fn=TextTSIrgcollate_fn)
    # assert len(dataloader) > 0


    datamodule = MIMIC3DataModule(
        file_path=str(DATA_PATH / "output_mimic3/pheno"),
        tt_max=24,
        bert_type="prajjwal1/bert-tiny",
        batch_size=2,
        max_length=512
    )
    batch = dict()
    for batch in datamodule.train_dataloader():
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
    model = UTDEModule(
        task="pheno",
        tt_max=24, num_labels=25)
    loss = model(
        x_ts=batch["ts"],  # type ignore
        x_ts_mask=batch["ts_mask"],
        ts_tt_list=batch["ts_tt"],
        reg_ts=batch["reg_ts"],
        labels=batch["label"],
    )
    print(loss)
