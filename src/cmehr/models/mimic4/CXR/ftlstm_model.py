import numpy as np
from PIL import Image
import math
import ot
import ipdb
from einops import rearrange
from typing import Optional, List
from torch import nn
from torch.nn import Parameter
from transformers import AutoModel
import torch.nn.functional as F
import torch
from cmehr.models.mimic4.UTDE_modules import multiTimeAttention
from cmehr.models.mimic4.base_model import MIMIC4LightningModule
# from cmehr.models.mimic3.base_model import MIMIC3NoteModule
# from cmehr.models.mimic4.UTDE_modules import BertForRepresentation
from cmehr.backbone.vision.pretrained import get_biovil_t_image_encoder


class FTLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, initializer_range=0.02, batch_first=True, bidirectional=True):
        super(FTLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.c1 = torch.Tensor([1]).float()
        self.c2 = torch.Tensor([np.e]).float()
        self.c3 = torch.Tensor([0.]).float()
        self.ones = torch.ones([1, self.hidden_size]).float()
        self.register_buffer('c1_const', self.c1)
        self.register_buffer('c2_const', self.c2)
        self.register_buffer('c3_const', self.c3)
        self.register_buffer("ones_const", self.ones)
        # Input Gate Parameter
        self.Wi = Parameter(torch.normal(0.0, initializer_range, size=(self.input_size, self.hidden_size)))
        self.Ui = Parameter(torch.normal(0.0, initializer_range, size=(self.hidden_size, self.hidden_size)))
        self.bi = Parameter(torch.zeros(self.hidden_size))
        # Forget Gate Parameter
        self.Wf = Parameter(torch.normal(0.0, initializer_range, size=(self.input_size, self.hidden_size)))
        self.Uf = Parameter(torch.normal(0.0, initializer_range, size=(self.hidden_size, self.hidden_size)))
        self.bf = Parameter(torch.zeros(self.hidden_size))
        # Output Gate Parameter
        self.Wog = Parameter(torch.normal(0.0, initializer_range, size=(self.input_size, self.hidden_size)))
        self.Uog = Parameter(torch.normal(0.0, initializer_range, size=(self.hidden_size, self.hidden_size)))
        self.bog = Parameter(torch.zeros(self.hidden_size))
        # Cell Layer Parameter
        self.Wc = Parameter(torch.normal(0.0, initializer_range, size=(self.input_size, self.hidden_size)))
        self.Uc = Parameter(torch.normal(0.0, initializer_range, size=(self.hidden_size, self.hidden_size)))
        self.bc = Parameter(torch.zeros(self.hidden_size))
        # Decomposition Layer Parameter
        self.W_decomp = Parameter(
            torch.normal(0.0, initializer_range, size=(self.hidden_size, self.hidden_size)))
        self.b_decomp = Parameter(torch.zeros(self.hidden_size))
        # Decay Parameter
        self.W_decay_1 = Parameter(torch.tensor([[0.33]]))
        self.W_decay_2 = Parameter(torch.tensor([[0.33]]))
        self.W_decay_3 = Parameter(torch.tensor([[0.33]]))
        self.a = Parameter(torch.tensor([1.0]))
        self.b = Parameter(torch.tensor([1.0]))
        self.m = Parameter(torch.tensor([0.02]))
        self.k = Parameter(torch.tensor([2.9]))
        self.d = Parameter(torch.tensor([4.5]))
        self.n = Parameter(torch.tensor([2.5]))

    def FTLSTM_unit(self, prev_hidden_memory, inputs, times):
        prev_hidden_state, prev_cell = prev_hidden_memory
        x = inputs
        t = times
        T = self.map_elapse_time(t)
        C_ST = torch.tanh(torch.matmul(prev_cell, self.W_decomp) + self.b_decomp)
        C_ST_dis = torch.mul(T, C_ST)
        prev_cell = prev_cell - C_ST + C_ST_dis

        # Input Gate
        i = torch.sigmoid(torch.matmul(x, self.Wi) +
                          torch.matmul(prev_hidden_state, self.Ui) + self.bi)
        # Forget Gate
        f = torch.sigmoid(torch.matmul(x, self.Wf) +
                          torch.matmul(prev_hidden_state, self.Uf) + self.bf)
        # Output Gate
        o = torch.sigmoid(torch.matmul(x, self.Wog) +
                          torch.matmul(prev_hidden_state, self.Uog) + self.bog)
        # Candidate Memory Cell
        C = torch.sigmoid(torch.matmul(x, self.Wc) +
                          torch.matmul(prev_hidden_state, self.Uc) + self.bc)
        # Current Memory Cell
        Ct = f * prev_cell + i * C

        # Current Hidden State
        current_hidden_state = o * torch.tanh(Ct)

        return current_hidden_state, Ct

    def map_elapse_time(self, t):
        T_1 = torch.div(self.c1_const, torch.mul(self.a, torch.pow(t, self.b)))
        T_2 = self.k - torch.mul(self.m, t)
        T_3 = torch.div(self.c1_const, (self.c1_const + torch.pow(torch.div(t, self.d), self.n)))
        T = torch.mul(self.W_decay_1, T_1) + torch.mul(self.W_decay_2, T_2) + torch.mul(self.W_decay_3, T_3)
        T = torch.max(T, self.c3_const)
        T = torch.min(T, self.c1_const)
        T = torch.matmul(T, self.ones_const)
        return T

    def forward(self, inputs, times):
        device = inputs.device
        if self.batch_first:
            batch_size = inputs.size()[0]
            inputs = inputs.permute(1, 0, 2)
            times = times.transpose(0, 1)
        else:
            batch_size = inputs.size()[1]
        # batch_size = inputs.size()[0]
        prev_hidden = torch.zeros((batch_size, self.hidden_size), device=device)
        prev_cell = torch.zeros((batch_size, self.hidden_size), device=device)
        seq_len = inputs.size()[0]
        hidden_his = []
        for i in range(seq_len):
            prev_hidden, prev_cell = self.FTLSTM_unit((prev_hidden, prev_cell), inputs[i], times[i])
            hidden_his.append(prev_hidden)
        hidden_his = torch.stack(hidden_his)
        if self.bidirectional:
            second_hidden = torch.zeros((batch_size, self.hidden_size), device=device)
            second_cell = torch.zeros((batch_size, self.hidden_size), device=device)
            second_inputs = torch.flip(inputs, [0])
            second_times = torch.flip(times, [0])
            second_hidden_his = []
            for i in range(seq_len):
                if i == 0:
                    time = times[i]
                else:
                    time = second_times[i-1]
                second_hidden, second_cell = self.FTLSTM_unit((second_hidden, second_cell), second_inputs[i], time)
                second_hidden_his.append(second_hidden)
            second_hidden_his = torch.stack(second_hidden_his)
            hidden_his = torch.cat((hidden_his, second_hidden_his), dim=2)
            prev_hidden = torch.cat((prev_hidden, second_hidden), dim=1)
            prev_cell = torch.cat((prev_cell, second_cell), dim=1)
        if self.batch_first:
            hidden_his = hidden_his.permute(1, 0, 2)
        return hidden_his, (prev_hidden, prev_cell)


class MIMIC4FTLSTMModule(MIMIC4LightningModule):
    def __init__(self,
                 task: str = "ihm",
                 modeltype: str = "TS",
                 max_epochs: int = 10,
                 ts_learning_rate: float = 4e-4,
                 period_length: int = 48,
                 embed_dim: int = 128,
                #  bert_type: str = "prajjwal1/bert-tiny",
                 dropout: float = 0.1,
                 *args,
                 **kwargs
                 ):
        super().__init__(task=task, modeltype=modeltype, max_epochs=max_epochs,
                         ts_learning_rate=ts_learning_rate, period_length=period_length)
        self.save_hyperparameters()

        self.img_encoder = get_biovil_t_image_encoder()
        self.img_embed_dim = 512
        self.img_proj = nn.Linear(self.img_embed_dim, embed_dim)

        self.d_txt = 128
        self.ftlstm = FTLSTM(self.d_txt,
                            embed_dim // 2,
                            batch_first=True,
                            bidirectional=True)
        self.proj1 = nn.Linear(self.d_txt, self.d_txt)
        self.proj2 = nn.Linear(self.d_txt, self.d_txt)
        self.out_layer = nn.Linear(self.d_txt, self.num_labels)
        self.dropout = dropout

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
        x_img = self.img_proj(x_img)

        valid_x_img = x_img[cxr_time_mask.bool()].unsqueeze(0)
        valid_cxr_time = cxr_time[cxr_time_mask.bool()].unsqueeze(0)

        # lstm_out, _ = self.ftlstm(x_img, cxr_time)
        lstm_out, _ = self.ftlstm(valid_x_img, valid_cxr_time)
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
    # from cmehr.dataset.mimic3_downstream_datamodule import MIMIC3DataModule
    from cmehr.dataset.mimic4_downstream_datamodule import MIMIC4DataModule

    datamodule = MIMIC4DataModule(
        mimic_cxr_dir=str(MIMIC_CXR_JPG_PATH),
        file_path=str(DATA_PATH / "output_mimic4/TS_CXR/pheno"),
        period_length=48,
        batch_size=1
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
    model = MIMIC4FTLSTMModule()
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

