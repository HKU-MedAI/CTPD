import torch
import torch.nn as nn
import torch.nn.functional as F

from cmehr.models.mimic4.base_model import MIMIC4LightningModule

from cmehr.backbone.time_series.inceptiontime import InceptionTimeFeatureExtractor
from cmehr.backbone.time_series.resnet import ResNetFeatureExtractor
from cmehr.backbone.time_series.pooling import MILConjunctivePooling, MILAdditivePooling

import ipdb


class MILLETModule(MIMIC4LightningModule):
    def __init__(self,
                 task: str = "ihm",
                 modeltype: str = "TS",
                 backbone: str = "resnet",
                 max_epochs: int = 10,
                 img_learning_rate: float = 1e-4,
                 ts_learning_rate: float = 4e-4,
                 period_length: int = 48,
                 orig_reg_d_ts: int = 30,
                 hidden_dim: int = 128,
                 n_layers: int = 3,
                 *args,
                 **kwargs):
        super().__init__(task=task, modeltype=modeltype, max_epochs=max_epochs,
                         img_learning_rate=img_learning_rate, ts_learning_rate=ts_learning_rate,
                         period_length=period_length)

        self.input_size = orig_reg_d_ts
        self.hidden_dim = hidden_dim

        # self.feature_extractor = InceptionTimeFeatureExtractor(orig_reg_d_ts, out_channels=hidden_dim // 4)
        self.feature_extractor = ResNetFeatureExtractor(n_in_channels=orig_reg_d_ts)
        dropout = 0.1
        apply_positional_encoding = True
        self.pool = MILConjunctivePooling(
            self.hidden_dim,
            self.num_labels,
            dropout=dropout,
            apply_positional_encoding=apply_positional_encoding,
        )

    def forward(self,
                reg_ts,
                labels=None,
                **kwargs):
        
        batch_size = reg_ts.size(0)
        x = reg_ts.permute(0, 2, 1)
        feat = self.feature_extractor(x)
        pool_output = self.pool(feat)
        output = pool_output["bag_logits"]

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
    model = MILLETModule(
    )
    loss = model(
        reg_ts=batch["reg_ts"],
        labels=batch["label"]
    )
    print(loss)
