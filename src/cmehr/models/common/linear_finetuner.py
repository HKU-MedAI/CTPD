import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn.metrics as metrics
from lightning import LightningModule


class LinearFinetuner(LightningModule):
    def __init__(self, 
                 in_size: int,
                 num_classes: int = 2,
                 model_type: str = "linear",
                 lr: float = 1e-3,
                 *args,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.in_size = in_size
        self.num_classes = num_classes
        if model_type == "linear":
            self.pred_layer = nn.Linear(self.in_size, self.num_classes)
        elif model_type == "mlp":
            self.pred_layer = nn.Sequential(
                nn.Linear(self.in_size, 512),
                nn.ReLU(),
                nn.Linear(512, num_classes)
            )

    def forward(self, x):
        return self.pred_layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss)
        return loss

    def on_validation_epoch_start(self) -> None:
        self.val_step_outputs = []

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)    
        step_output = {
            "logits": logits,
            "y": y
        }
        self.val_step_outputs.append(step_output)

    def on_validation_epoch_end(self) -> None:
        if self.num_classes == 2:
            logits = torch.cat([x["logits"] for x in self.val_step_outputs], dim=0).detach().cpu().numpy()
            y = torch.cat([x["y"] for x in self.val_step_outputs], dim=0).detach().cpu().numpy()
            auroc = metrics.roc_auc_score(y, logits[:, 1])
            auprc = metrics.average_precision_score(y, logits[:, 1])
            f1 = metrics.f1_score(y, logits.argmax(axis=1))
            metrics_dict = {
                "val_auroc": auroc,
                "val_auprc": auprc,
                "val_f1": f1
            }
            self.log_dict(metrics_dict, on_epoch=True, on_step=False,
                        batch_size=len(y))
        else:
            raise NotImplementedError("Multi-class classification is not supported yet.")

    def on_test_epoch_start(self) -> None:
        self.test_step_outputs = []

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        step_output = {
            "logits": logits,
            "y": y
        }
        self.test_step_outputs.append(step_output)

    def on_test_epoch_end(self) -> None:
        if self.num_classes == 2:
            logits = torch.cat([x["logits"] for x in self.test_step_outputs], dim=0).detach().cpu().numpy()
            y = torch.cat([x["y"] for x in self.test_step_outputs], dim=0).detach().cpu().numpy()
            auroc = metrics.roc_auc_score(y, logits[:, 1])
            auprc = metrics.average_precision_score(y, logits[:, 1])
            f1 = metrics.f1_score(y, logits.argmax(axis=1))
            metrics_dict = {
                "test_auroc": auroc,
                "test_auprc": auprc,
                "test_f1": f1
            }
            self.log_dict(metrics_dict, on_epoch=True, on_step=False,
                        batch_size=len(y))
        else:
            raise NotImplementedError("Multi-class classification is not supported yet.")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.4, patience=3, verbose=True, mode='max')
        scheduler = {
            'scheduler': lr_scheduler,
            'monitor': 'val_auroc',
            'interval': 'epoch',
            'frequency': 1
        }
        return [optimizer], [scheduler]
    