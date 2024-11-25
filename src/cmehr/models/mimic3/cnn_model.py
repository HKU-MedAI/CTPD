# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # from cmehr.models.mimic4.base_model import MIMIC3LightningModule
# from cmehr.models.mimic3.base_model import MIMIC3NoteModule

# import ipdb


# class CNNNoteModule(MIMIC3NoteModule):
#     def __init__(self,
#                  task: str = "ihm",
#                  modeltype: str = "TS",
#                  max_epochs: int = 10,
#                  ts_learning_rate: float = 4e-4,
#                  period_length: int = 48,
#                  orig_reg_d_ts: int = 30,
#                  hidden_dim: int = 128,
#                  n_layers: int = 3,
#                  *args,
#                  **kwargs):
#         super().__init__(task=task, modeltype=modeltype, max_epochs=max_epochs,
#                          ts_learning_rate=ts_learning_rate,
#                          period_length=period_length)

#         self.input_size = orig_reg_d_ts
#         self.n_layers = n_layers
#         self.hidden_dim = hidden_dim

#         self.conv_block1 = nn.Sequential(
#             nn.Conv1d(self.input_size, 32, kernel_size=3,
#                       padding=1),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2, stride=2),
#         )

#         self.conv_block2 = nn.Sequential(
#             nn.Conv1d(32, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2, stride=2),
#         )

#         self.conv_block3 = nn.Sequential(
#             nn.Conv1d(64, self.hidden_dim, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2, stride=2),
#             nn.Dropout(0.3)
#         )

#         self.fc = nn.Linear(self.hidden_dim * (self.tt_max // 8), self.num_labels)

#     def forward_feat(self, x):
#         x = x.permute(0, 2, 1)
#         x = self.conv_block1(x) # (B, 30, 48 -> B, 32, 24)
#         x = self.conv_block2(x) # (B, 32, 24 -> B, 64, 12)
#         x = self.conv_block3(x) # (B, 64, 12 -> B, 128, 6)
#         return x
    
#     def forward(self,
#                 reg_ts,
#                 labels=None,
#                 **kwargs):

#         x = reg_ts
#         batch_size = x.size(0)
#         feat = self.forward_feat(x)
#         feat = feat.view(batch_size, -1) # (B, 128, 6 -> B, 128*6)
#         output = self.fc(feat)

#         if self.task in ['ihm', 'readm']:
#             if labels != None:
#                 ce_loss = self.loss_fct1(output, labels)
#                 return ce_loss
#             return F.softmax(output, dim=-1)[:, 1]

#         elif self.task == 'pheno':
#             if labels != None:
#                 labels = labels.float()
#                 ce_loss = self.loss_fct1(output, labels)
#                 return ce_loss
#             return torch.sigmoid(output)


# if __name__ == "__main__":
#     from torch.utils.data import DataLoader
#     from cmehr.paths import *
#     from cmehr.dataset.mimic3_downstream_datamodule import MIMIC3DataModule

#     datamodule = MIMIC3DataModule(
#         file_path=str(DATA_PATH / "output_mimic3/ihm"),
#         tt_max=48
#     )
#     for batch in datamodule.val_dataloader():
#         break
#     model = CNNNoteModule(
#         task="ihm",
#         period_length=48)
#     loss = model(
#         reg_ts=batch["reg_ts"],
#         labels=batch["label"]
#     )
#     print(loss)
