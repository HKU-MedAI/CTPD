import torch
import torch.nn as nn
import torch.nn.functional as F

from cmehr.models.mimic4.base_model import MIMIC4LightningModule

import ipdb


class RNNModule(MIMIC4LightningModule):
    def __init__(self,
                 task: str = "ihm",
                 modeltype: str = "TS",
                 max_epochs: int = 10,
                 img_learning_rate: float = 1e-4,
                 ts_learning_rate: float = 4e-4,
                 period_length: int = 48,
                 orig_reg_d_ts: int = 30,
                 hidden_dim=128,
                 n_layers=3,
                 *args,
                 **kwargs):
        super().__init__(task=task, modeltype=modeltype, max_epochs=max_epochs,
                         img_learning_rate=img_learning_rate, ts_learning_rate=ts_learning_rate,
                         period_length=period_length)

        self.input_size = orig_reg_d_ts
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # define an RNN with specified parameters
        self.rnn = nn.RNN(self.input_size, self.hidden_dim,
                          self.n_layers, batch_first=True)

        # last, fully-connected layers
        self.fc = nn.Linear(hidden_dim, self.num_labels)

    # def init_hidden(self, batch_size):
    #     hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
    #     print('hidden: ', hidden.shape)
    #     return hidden

    def forward(self,
                reg_ts,
                labels=None,
                **kwargs):
        
        # x (batch_size, seq_length, input_size)
        # hidden (n_layers, batch_size, hidden_dim)
        # r_out (batch_size, time_step, hidden_size)
        batch_size = reg_ts.size(0)

        # hidden = self.init_hidden(batch_size)
        # get RNN outputs
        # r_out, hidden = self.rnn(reg_ts, hidden)
        r_out, _ = self.rnn(reg_ts)
        # print('r_out: ', r_out.shape)
        # shape output to be (batch_size*seq_length, hidden_dim)
        r_out = r_out[:, -1, :]
        # print('r_out: ', r_out.shape)

        # get final output
        output = self.fc(r_out)
        # print('output: ', output.shape)

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
        modeltype="TS",
        tt_max=48
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
    model = RNN(
        use_multiscale=False,
        use_prototype=False
    )
    loss = model(
        x=batch["reg_ts"],
        labels=None
    )
    print(loss)
