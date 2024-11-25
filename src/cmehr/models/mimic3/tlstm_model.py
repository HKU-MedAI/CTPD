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


class TimeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, cuda_flag=True, bidirectional=False):
        # assumes that batch_first is always true
        super(TimeLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.cuda_flag = cuda_flag
        self.W_all = nn.Linear(hidden_size, hidden_size * 4)
        self.U_all = nn.Linear(input_size, hidden_size * 4)
        self.W_d = nn.Linear(hidden_size, hidden_size)
        self.bidirectional = bidirectional

    def forward(self, inputs, timestamps, reverse=False):
        # inputs: [b, seq, embed]
        # h: [b, hid]
        # c: [b, hid]
        b, seq, embed = inputs.size()
        h = torch.zeros(b, self.hidden_size, requires_grad=False)
        c = torch.zeros(b, self.hidden_size, requires_grad=False)
        if self.cuda_flag:
            h = h.cuda()
            c = c.cuda()
        outputs = []
        for s in range(seq):
            c_s1 = torch.tanh(self.W_d(c))
            c_s2 = c_s1 * timestamps[:, s:s + 1].expand_as(c_s1)
            c_l = c - c_s1
            c_adj = c_l + c_s2
            outs = self.W_all(h) + self.U_all(inputs[:, s])
            f, i, o, c_tmp = torch.chunk(outs, 4, 1)
            f = torch.sigmoid(f)
            i = torch.sigmoid(i)
            o = torch.sigmoid(o)
            c_tmp = torch.sigmoid(c_tmp)
            c = f * c_adj + i * c_tmp
            h = o * torch.tanh(c)
            outputs.append(h)
        if reverse:
            outputs.reverse()
        outputs = torch.stack(outputs, 1)
        return outputs


class TLSTMModule(MIMIC3NoteModule):
    def __init__(self,
                 task: str = "ihm",
                 modeltype: str = "TS",
                 max_epochs: int = 10,
                 ts_learning_rate: float = 4e-4,
                 period_length: int = 48,
                 embed_dim: int = 128,
                 bert_type: str = "prajjwal1/bert-tiny",
                 dropout: float = 0.1,
                 *args,
                 **kwargs
                 ):
        super().__init__(task=task, modeltype=modeltype, max_epochs=max_epochs,
                         ts_learning_rate=ts_learning_rate, period_length=period_length)
        self.save_hyperparameters()

        Biobert = AutoModel.from_pretrained(bert_type)
        self.bertrep = BertForRepresentation(bert_type, Biobert)
        self.d_txt = 128
        self.tlstm = TimeLSTM(self.d_txt, embed_dim)
        self.proj1 = nn.Linear(self.d_txt, self.d_txt)
        self.proj2 = nn.Linear(self.d_txt, self.d_txt)
        self.out_layer = nn.Linear(self.d_txt, self.num_labels)
        self.dropout = dropout

    def forward(self,
                input_ids_sequences=None,
                attn_mask_sequences=None,
                note_time_list=None,
                note_time_mask_list=None,
                labels=None, reg_ts=None,
                **kwargs):
        
        x_txt = self.bertrep(input_ids_sequences, attn_mask_sequences)
        lstm_out = self.tlstm(x_txt, note_time_list)
        last_hs = lstm_out[:, -1, :]
        # last_hs = torch.mean(x_txt, dim=1)
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
    from cmehr.dataset.mimic3_downstream_datamodule import MIMIC3DataModule

    datamodule = MIMIC3DataModule(
        file_path=str(DATA_PATH / "output_mimic3/readm"),
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
    model = TLSTMModule(
        task="readm",
        period_length=48,
        bert_type="prajjwal1/bert-tiny",
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
