import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_pretrained_bert as Bert
from cmehr.models.mimic4.base_model import MIMIC4LightningModule

import ipdb


class PatchEmbed(nn.Module):
    def __init__(self, seq_len, patch_size=4, in_chans=30, embed_dim=128):
        super().__init__()
        stride = max(1, patch_size // 2)
        num_patches = int((seq_len - patch_size) / stride + 1)
        self.num_patches = num_patches
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)

    def forward(self, x):
        x_out = self.proj(x).flatten(2).transpose(1, 2)
        return x_out


class TransformerModule(MIMIC4LightningModule):
    def __init__(self,
                 task: str = "ihm",
                 modeltype: str = "TS",
                 max_epochs: int = 10,
                 img_learning_rate: float = 1e-4,
                 ts_learning_rate: float = 4e-4,
                 period_length: int = 48,
                 orig_reg_d_ts: int = 30,
                 hidden_dim=512,
                 n_layers=3,
                 *args,
                 **kwargs):
        super().__init__(task=task, modeltype=modeltype, max_epochs=max_epochs,
                         img_learning_rate=img_learning_rate, ts_learning_rate=ts_learning_rate,
                         period_length=period_length)

        self.input_size = orig_reg_d_ts
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.patch_size = 4
        self.dropout_rate = 0.15
        self.patch_embed = PatchEmbed(
            seq_len=self.tt_max, patch_size=self.patch_size,
            in_chans=self.input_size, embed_dim=self.hidden_dim
        )
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.hidden_dim), requires_grad=True)
        self.pos_drop = nn.Dropout(p=self.dropout_rate)

        config = Bert.modeling.BertConfig(
            vocab_size_or_config_json_file=100, # it doesn't matter
            num_hidden_layers=2,
            hidden_size=self.hidden_dim,
            num_attention_heads=8
        )
        self.encoder = Bert.modeling.BertEncoder(config=config)
        self.fc = nn.Linear(hidden_dim, self.num_labels)

    def forward(self,
                reg_ts,
                labels=None,
                **kwargs):
        
        batch_size = reg_ts.size(0)
        # patchify time series
        x = reg_ts.transpose(1, 2)
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        attention_mask = torch.ones(x.shape[:-1]).type_as(reg_ts)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        encoded_layers = self.encoder(x, extended_attention_mask)
        sequence_output = encoded_layers[-1]
        pooled_output = sequence_output.mean(dim=1)
        output = self.fc(pooled_output)

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
        file_path=str(ROOT_PATH / "output_mimic4/TS_CXR/ihm"),
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
    model = TransformerModule(
    )
    loss = model(
        reg_ts=batch["reg_ts"],
        labels=batch["label"]
    )
    print(loss)
