import math
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter
import ipdb
from cmehr.models.mimic4.base_model import MIMIC4LightningModule


class FilterLinear(nn.Module):
    def __init__(self, in_features, out_features, filter_square_matrix, bias=True):
        '''
        filter_square_matrix : filter square matrix, whose each elements is 0 or 1.
        '''
        super(FilterLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        assert in_features > 1 and out_features > 1, "Passing in nonsense sizes"

        self.filter_square_matrix = None
        self.filter_square_matrix = Variable(
            filter_square_matrix, requires_grad=False)

        self.weight = Parameter(torch.Tensor(out_features, in_features))

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        self.filter_square_matrix = self.filter_square_matrix.type_as(x)
        return F.linear(
            x,
            self.filter_square_matrix.mul(self.weight),
            self.bias
        )

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', bias=' + str(self.bias is not None) + ')'


class GRUDModule(MIMIC4LightningModule):
    def __init__(self,
                 task: str = "ihm",
                 modeltype: str = "TS",
                 max_epochs: int = 10,
                 img_learning_rate: float = 1e-4,
                 ts_learning_rate: float = 4e-4,
                 period_length: int = 48,
                 input_size: int = 17,
                 hidden_size: int = 15,
                 dropout: float = 0.1,
                 num_layers: int = 49,
                 x_mean: torch.Tensor = torch.tensor(0.),
                 bias: bool = True,
                 batch_first: bool = False,
                 bidirectional: bool = False,
                 dropout_type: str = "mloss",
                 static: bool = True,
                 batch_size: int = 128,
                 *args,
                 **kwargs
                 ):
        super().__init__(task=task, modeltype=modeltype, max_epochs=max_epochs,
                         img_learning_rate=img_learning_rate, ts_learning_rate=ts_learning_rate,
                         period_length=period_length)
        self.save_hyperparameters()

        self.hidden_size = hidden_size
        self.delta_size = input_size
        self.mask_size = input_size

        # self.X_mean = 0.

        # Wz, Uz are part of the same network. the bias is bz
        self.zl = nn.Linear(input_size + hidden_size +
                            self.mask_size, hidden_size)
        # Wr, Ur are part of the same network. the bias is br
        self.rl = nn.Linear(input_size + hidden_size +
                            self.mask_size, hidden_size)
        # W, U are part of the same network. the bias is b
        self.hl = nn.Linear(input_size + hidden_size +
                            self.mask_size, hidden_size)

        self.identity = torch.eye(self.delta_size)
        self.gamma_x_l = FilterLinear(
            self.delta_size, self.delta_size, self.identity)

        # this was wrong in available version. remember to raise the issue
        self.gamma_h_l = nn.Linear(self.delta_size, self.hidden_size)

        # self.output_last = output_last

        self.fc = nn.Linear(self.hidden_size, 2)
        self.bn = torch.nn.BatchNorm1d(2, eps=1e-05, momentum=0.1, affine=True)
        self.drop = nn.Dropout(p=0.5, inplace=False)

        self.pred_layer = nn.Linear(self.hidden_size, self.num_labels)

    def step(self, x, x_last_obsv, h, mask, delta):
        """
        Inputs:
            x: input tensor
            x_last_obsv: input tensor with forward fill applied
            x_mean: the mean of each feature
            h: the hidden state of the network
            mask: the mask of whether or not the current value is observed
            delta: the tensor indicating the number of steps since the last time a feature was observed.

        Returns:
            h: the updated hidden state of the network
        """

        batch_size = x.size()[0]
        dim_size = x.size()[1]

        self.identity = self.identity.type_as(x)
        self.zeros = torch.zeros(batch_size, self.delta_size).type_as(x)
        self.zeros_h = torch.zeros(batch_size, self.hidden_size).type_as(x)

        gamma_x_l_delta = self.gamma_x_l(delta)
        # exponentiated negative rectifier
        delta_x = torch.exp(-torch.max(self.zeros, gamma_x_l_delta))

        gamma_h_l_delta = self.gamma_h_l(delta)
        # self.zeros became self.zeros_h to accomodate hidden size != input size
        delta_h = torch.exp(-torch.max(self.zeros_h, gamma_h_l_delta))

        # x_mean = x_mean.repeat(batch_size, 1)
        x = mask * x + (1 - mask) * (delta_x *
                                     x_last_obsv + (1 - delta_x) * 0.)
        h = delta_h * h
        combined = torch.cat((x, h, mask), 1)
        # sigmoid(W_z*x_t + U_z*h_{t-1} + V_z*m_t + bz)
        z = torch.sigmoid(self.zl(combined))
        # sigmoid(W_r*x_t + U_r*h_{t-1} + V_r*m_t + br)
        r = torch.sigmoid(self.rl(combined))
        combined_new = torch.cat((x, r*h, mask), 1)
        # tanh(W*x_t +U(r_t*h_{t-1}) + V*m_t) + b
        h_tilde = torch.tanh(self.hl(combined_new))
        h = (1 - z) * h + z * h_tilde

        return h

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
        # compute delta
        x_delta = torch.zeros_like(x_ts_mask).type_as(x_ts)
        x_last_obs = torch.zeros_like(x_ts).type_as(x_ts)
        for i in range(x_ts_mask.shape[0]):
            # for each feature
            for j in range(x_ts_mask.shape[2]):
                for k in range(x_ts_mask.shape[1]):
                    if k == 0:
                        x_delta[i, k, j] = 0
                    else:
                        delta_time = ts_tt_list[i, k] - ts_tt_list[i, k-1]
                        if delta_time > 0:
                            if x_ts_mask[i, k, j] == 1:
                                x_delta[i, k, j] = delta_time + \
                                    x_delta[i, k-1, j]
                                x_last_obs[i, k, j] = x_ts[i, k, j]
                            else:
                                x_delta[i, k, j] = delta_time
                                x_last_obs[i, k, j] = x_last_obs[i, k-1, j]
        x_delta /= x_delta.max()

        X = x_ts
        X_last_obsv = x_last_obs
        Mask = x_ts_mask
        Delta = x_delta

        batch_size = X.size(0)
        step_size = X.size(1)  # num timepoints
        spatial_size = X.size(2)  # num features
        Hidden_State = self.initHidden(batch_size)

        outputs = None
        for i in range(step_size):
            Hidden_State = self.step(
                torch.squeeze(X[:, i:i+1, :], 1),
                torch.squeeze(X_last_obsv[:, i:i+1, :], 1),
                # torch.squeeze(self.X_mean[:, i:i+1, :], 1),
                Hidden_State.type_as(X),
                torch.squeeze(Mask[:, i:i+1, :], 1),
                torch.squeeze(Delta[:, i:i+1, :], 1),
            )
            if outputs is None:
                outputs = Hidden_State.unsqueeze(1)
            else:
                outputs = torch.cat((Hidden_State.unsqueeze(1), outputs), 1)

        logits = self.pred_layer(outputs[:, -1, :])
        if self.task in ['ihm', 'readm']:
            if labels != None:
                ce_loss = self.loss_fct1(logits, labels)
                return ce_loss
            return F.softmax(logits, dim=-1)[:, 1]

        elif self.task == 'pheno':
            if labels != None:
                labels = labels.float()
                ce_loss = self.loss_fct1(logits, labels)
                return ce_loss
            return torch.sigmoid(logits)

    def initHidden(self, batch_size):
        Hidden_State = torch.zeros(batch_size, self.hidden_size)
        return Hidden_State


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
    model = GRUDModule(
        batch_size=4
    )
    loss = model(
        x_ts=batch["ts"],  # type ignore
        x_ts_mask=batch["ts_mask"],
        ts_tt_list=batch["ts_tt"],
        reg_ts=batch["reg_ts"],
        labels=batch["label"],
    )
    print(loss)
