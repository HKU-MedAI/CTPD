import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
import numpy as np
import ipdb
from cmehr.models.mimic4.base_model import MIMIC4LightningModule


class PositionalEncodingTF(nn.Module):
    def __init__(self, d_model, max_len=500, MAX=10000,):
        super(PositionalEncodingTF, self).__init__()
        self.max_len = max_len
        self.d_model = d_model
        self.MAX = MAX
        self._num_timescales = d_model // 2

    def getPE(self, P_time):
        B = P_time.shape[1]

        P_time = P_time.float()

        timescales = self.max_len ** np.linspace(0, 1, self._num_timescales)

        times = torch.Tensor(P_time.cpu()).unsqueeze(2)

        scaled_time = times / torch.Tensor(timescales[None, None, :])
        # Use a 32-D embedding to represent a single time point
        pe = torch.cat([torch.sin(scaled_time), torch.cos(
            scaled_time)], axis=-1)  # T x B x d_model
        pe = pe.type(torch.FloatTensor)

        return pe

    def forward(self, P_time):
        pe = self.getPE(P_time)
        return pe


class SEFTModule(MIMIC4LightningModule):
    def __init__(self,
                 task: str = "ihm",
                 modeltype: str = "TS",
                 max_epochs: int = 10,
                 img_learning_rate: float = 1e-4,
                 ts_learning_rate: float = 4e-4,
                 period_length: int = 48,
                 d_inp: int = 13,
                 d_model: int = 13,
                 nhead: int = 2,
                 nhid: int = 30,
                 nlayers: int = 2,
                 dropout: float = 0.2,
                 max_len: int = 472,
                 d_static: int = 2,
                 MAX: int = 100,
                 perc: float = 0.5,
                 aggreg: str = "mean",
                 static=True,
                 *args,
                 **kwargs
                 ):
        super().__init__(task=task, modeltype=modeltype, max_epochs=max_epochs,
                         img_learning_rate=img_learning_rate, ts_learning_rate=ts_learning_rate,
                         period_length=period_length)
        self.save_hyperparameters()

        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        d_pe = 16
        d_enc = d_inp

        self.pos_encoder = PositionalEncodingTF(d_pe, max_len, MAX)
        self.pos_encoder_value = PositionalEncodingTF(d_pe, max_len, MAX)
        self.pos_encoder_sensor = PositionalEncodingTF(d_pe, max_len, MAX)

        self.linear_value = nn.Linear(1, 16)
        self.linear_sensor = nn.Linear(1, 16)

        self.d_K = 2 * (d_pe + 16+16)

        encoder_layers = TransformerEncoderLayer(self.d_K, 1, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        encoder_layers_f_prime = TransformerEncoderLayer(
            int(self.d_K//2), 1, nhid, dropout)
        self.transformer_encoder_f_prime = TransformerEncoder(
            encoder_layers_f_prime, 2)

        self.emb = nn.Linear(d_static, 16)

        self.proj_weight = nn.Parameter(torch.Tensor(self.d_K, 128))

        self.lin_map = nn.Linear(self.d_K, 128)
        d_fi = 128 + 16

        if static == False:
            d_fi = 128
        else:
            d_fi = 128 + d_pe
        self.mlp = nn.Sequential(
            nn.Linear(d_fi, d_fi),
            nn.ReLU(),
            nn.Linear(d_fi, self.num_labels),
        )

        self.aggreg = aggreg

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

        import json
        with open("/home/*/Documents/MMMSPG/src/cmehr/preprocess/mimic3/mimic3models/resources/discretizer_config.json", "r") as f:
            config = json.load(f)
        variables = config["id_to_channel"]
        static_variables = ["Height", "Weight"]
        inp_variables = list(set(variables) - set(static_variables))
        self.static_indices = [variables.index(v) for v in static_variables]
        self.inp_indices = [variables.index(v) for v in inp_variables]

    def init_weights(self):
        initrange = 1e-10
        self.emb.weight.data.uniform_(-initrange, initrange)
        self.linear_value.weight.data.uniform_(-initrange, initrange)
        self.linear_sensor.weight.data.uniform_(-initrange, initrange)
        self.lin_map.weight.data.uniform_(-initrange, initrange)
        xavier_uniform_(self.proj_weight)

    def forward(self, x_ts, x_ts_mask, ts_tt_list,
                cxr_imgs_sequences=None,
                cxr_time_sequences=None,
                cxr_time_mask_sequence=None,
                labels=None, reg_ts=None):
        """ Forward function of Multimodal model
        :param x_ts: (B, N, D_t), torch.Tensor, time series data
        :param x_ts_mask: (B, N, D_t), torch.Tensor, time series mask
        :param ts_tt_list: (B, N), torch.Tensor, time series time
        :param labels: (B, ), torch.Tensor, labels
        :param reg_ts: (B, N_r, D_r), torch.Tensor, regular time series data
        """
        times = ts_tt_list
        src = x_ts[:, :, self.inp_indices]
        static = x_ts[:, :, self.static_indices]
        maxlen, batch_size = src.shape[1], src.shape[0]
        src_mask = x_ts_mask[:, :, self.inp_indices]

        src = torch.cat([src, src_mask], dim=2)
        static = torch.mean(static, dim=1)

        fea = src[:, :, :int(src.shape[2]/2)]

        output = torch.zeros((batch_size, self.d_K)).type_as(x_ts)
        for i in range(batch_size):
            nonzero_index = fea[i].nonzero(as_tuple=False)
            if nonzero_index.shape[0] == 0:
                continue
            values = fea[i][nonzero_index[:, 0],
                            nonzero_index[:, 1]]  # v in SEFT paper
            time_index = nonzero_index[:, 0]
            time_sequence = times[i]
            time_points = time_sequence[time_index]  # t in SEFT paper
            pe_ = self.pos_encoder(
                time_points.unsqueeze(1)).squeeze(1).type_as(x_ts)

            # the dimensions of variables. The m value in SEFT paper.
            variable = nonzero_index[:, 1]
            unit = torch.cat([pe_, values.unsqueeze(
                1), variable.unsqueeze(1)], dim=1)

            variable_ = self.pos_encoder_sensor(
                variable.unsqueeze(1)).squeeze(1).type_as(x_ts)

            values_ = self.linear_value(values.float().unsqueeze(1)).squeeze(1)

            unit = torch.cat([pe_, values_, variable_], dim=1)

            f_prime = torch.mean(unit, dim=0)

            x = torch.cat([f_prime.repeat(unit.shape[0], 1), unit], dim=1)

            x = x.unsqueeze(1)

            output_unit = x
            output_unit = torch.mean(output_unit, dim=0)
            output[i, :] = output_unit

        output = self.lin_map(output)

        if static is not None:
            emb = self.emb(static)

        # feed through MLP
        if static is not None:
            output = torch.cat([output, emb], dim=1)

        output = self.mlp(output)

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
    from cmehr.dataset.mimic4_pretraining_datamodule import MIMIC4DataModule

    datamodule = MIMIC4DataModule(
        file_path=str(ROOT_PATH / "output_mimic4/ihm"),
        tt_max=48,
        batch_size=4
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
    model = SEFTModule(
    )
    loss = model(
        x_ts=batch["ts"],  # type ignore
        x_ts_mask=batch["ts_mask"],
        ts_tt_list=batch["ts_tt"],
        reg_ts=batch["reg_ts"],
        labels=batch["label"],
    )
    print(loss)
