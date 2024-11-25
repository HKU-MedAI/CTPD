import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ipdb
from cmehr.models.mimic4.base_model import MIMIC4LightningModule


class SingleChannelInterp(nn.Module):
    def __init__(self, ref_points, hours_look_ahead, d_dim):
        super(SingleChannelInterp, self).__init__()
        self.ref_points = ref_points
        self.hours_look_ahead = hours_look_ahead
        self.d_dim = d_dim // 4
        self.kernel = nn.Parameter(torch.zeros(self.d_dim))
        self.activation = nn.Sigmoid()

    def forward(self, x, reconstruction=False):
        self.time_stamp = x.shape[2]

        self.reconstruction = reconstruction
        x_t = x[:, :self.d_dim, :]
        # time
        d = x[:, 2*self.d_dim:3*self.d_dim, :]
        if reconstruction:
            output_dim = self.time_stamp
            m = x[:, 3*self.d_dim:, :]
            ref_t = d.unsqueeze(2).repeat(1, 1, output_dim, 1)
        else:
            m = x[:, self.d_dim: 2*self.d_dim, :]
            ref_t = torch.linspace(
                0, self.hours_look_ahead, self.ref_points).unsqueeze(0).type_as(x)
            ref_t /= self.hours_look_ahead
            output_dim = self.ref_points
        d = d.unsqueeze(3).repeat(1, 1, 1, output_dim)
        mask = m.unsqueeze(3).repeat(1, 1, 1, output_dim)
        x_t = x_t.unsqueeze(3).repeat(1, 1, 1, output_dim)
        norm = (d - ref_t) ** 2
        a = torch.ones((self.d_dim, self.time_stamp, output_dim)).type_as(x)
        pos_kernel = torch.log(1 + torch.exp(self.kernel))
        alpha = a * pos_kernel.unsqueeze(1).unsqueeze(2)
        # w = torch.logsumexp(-alpha * norm + torch.log(mask), dim=2)
        w = torch.log(
            torch.sum(torch.exp(-alpha * norm + torch.log(mask)), dim=2) + 1e-6)
        w1 = w.unsqueeze(2).repeat(1, 1, self.time_stamp, 1)
        w1 = torch.exp(-alpha * norm + torch.log(mask) - w1)
        y = torch.sum(w1 * x_t, dim=2)
        if reconstruction:
            rep1 = torch.cat((y, w), dim=1)
        else:
            w_t = torch.log(
                torch.sum(torch.exp(-10.0 * alpha * norm +
                                    torch.log(mask)), dim=2) + 1e-6)
            w_t = w_t.unsqueeze(2).repeat(1, 1, self.time_stamp, 1)
            w_t = torch.exp(-10.0 * alpha * norm + torch.log(mask) - w_t)
            y_trans = torch.sum(w_t * x_t, dim=2)
            rep1 = torch.cat((y, w, y_trans), dim=1)

        return rep1

    # def compute_output_shape(self, input_shape):
    #     if self.reconstruction:
    #         return (input_shape[0], 2 * self.d_dim, self.time_stamp)
    #     return (input_shape[0], 3 * self.d_dim, self.ref_points)


class CrossChannelInterp(nn.Module):
    def __init__(self, d_dim):
        super(CrossChannelInterp, self).__init__()
        self.d_dim = d_dim // 3
        self.cross_channel_interp = nn.Parameter(torch.eye(self.d_dim))
        self.activation = nn.Sigmoid()

    def forward(self, x, reconstruction=False):
        self.reconstruction = reconstruction
        self.output_dim = x.shape[-1]
        y = x[:, :self.d_dim, :]
        w = x[:, self.d_dim:2*self.d_dim, :]
        intensity = torch.exp(w)
        y = y.transpose(1, 2)
        w = w.transpose(1, 2)
        w2 = w
        w = w.unsqueeze(3).repeat(1, 1, 1, self.d_dim)
        den = torch.logsumexp(w, dim=2)
        w = torch.exp(w2 - den)
        mean = torch.mean(y, dim=1)
        mean = mean.unsqueeze(1).repeat(1, self.output_dim, 1)
        w2 = torch.matmul(w * (y - mean), self.cross_channel_interp) + mean
        rep1 = w2.transpose(1, 2)
        if not reconstruction:
            y_trans = x[:, 2*self.d_dim:3*self.d_dim, :]
            y_trans = y_trans - rep1
            rep1 = torch.cat((rep1, intensity, y_trans), dim=1)
        return rep1

    def compute_output_shape(self, input_shape):
        if self.reconstruction:
            return (input_shape[0], self.d_dim, self.output_dim)
        return (input_shape[0], 3 * self.d_dim, self.output_dim)


class IPNetModule(MIMIC4LightningModule):
    def __init__(self,
                 task: str = "ihm",
                 modeltype: str = "TS",
                 max_epochs: int = 10,
                 img_learning_rate: float = 1e-4,
                 ts_learning_rate: float = 4e-4,
                 period_length: int = 48,
                 *args,
                 **kwargs
                 ):
        super().__init__(task=task, modeltype=modeltype, max_epochs=max_epochs,
                         img_learning_rate=img_learning_rate, ts_learning_rate=ts_learning_rate,
                         period_length=period_length)
        self.save_hyperparameters()

        self.sci = SingleChannelInterp(
            ref_points=192,
            hours_look_ahead=period_length,
            d_dim=17*4
        )
        self.cci = CrossChannelInterp(d_dim=45)

        self.gru = nn.GRU(45, 100, batch_first=True)

        self.pred_layer = nn.Linear(100, self.num_labels)

    def hold_out(self, mask, perc=0.2):
        """To implement the autoencoder component of the loss, we introduce a set
        of masking variables mr (and mr1) for each data point. If drop_mask = 0,
        then we removecthe data point as an input to the interpolation network,
        and includecthe predicted value at this time point when assessing
        the autoencoder loss. In practice, we randomly select 20% of the
        observed data points to hold out from
        every input time series."""
        drop_mask = torch.zeros_like(mask)
        drop_mask *= mask
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                count = int(torch.sum(mask[i, j]).item())
                if int(0.20*count) > 1:
                    index = 0
                    r = torch.ones(count, 1).type_as(mask)
                    b = torch.randperm(count)[:int(0.20*count)]
                    r[b] = 0
                    for k in range(mask.shape[2]):
                        if mask[i, j, k] > 0:
                            drop_mask[i, j, k] = r[index]
                            index += 1
        return drop_mask

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
        # sin_interp = self.sci(x_ts.permute(0, 2, 1))
        hold_out_mask = self.hold_out(x_ts_mask)
        ts_full = ts_tt_list.unsqueeze(2).repeat(1, 1, x_ts.shape[2])
        x_input = torch.cat([x_ts, x_ts_mask, ts_full, hold_out_mask], dim=2)
        x_input = torch.permute(x_input, (0, 2, 1))
        in_interp = self.sci(x_input)
        interp = self.cci(in_interp)

        in_interp_reconst = self.sci(x_input, reconstruction=True)
        interp_reconst = self.cci(in_interp, reconstruction=True)

        _, out = self.gru(interp.permute(0, 2, 1))
        output = self.pred_layer(out.squeeze(0))

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
    model = IPNetModule(
    )
    loss = model(
        x_ts=batch["ts"],  # type ignore
        x_ts_mask=batch["ts_mask"],
        ts_tt_list=batch["ts_tt"],
        reg_ts=batch["reg_ts"],
        labels=batch["label"],
    )
    print(loss)
