from typing import Dict
import ipdb
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from cmehr.models.mimic3.base_model import MIMIC3LightningModule


class MIMIC4LightningModule(MIMIC3LightningModule):
    '''
    Base lightning model on MIMIC IV dataset.
    '''

    def __init__(self,
                 task: str = "ihm",
                 modeltype: str = "TS_CXR",
                 max_epochs: int = 10,
                 img_learning_rate: float = 1e-4,
                 ts_learning_rate: float = 4e-4,
                 period_length: int = 48,
                 num_labels: int = 2,
                 *args,
                 **kwargs
                 ):

        super().__init__(task=task, modeltype=modeltype, max_epochs=max_epochs,
                         img_learning_rate=img_learning_rate, ts_learning_rate=ts_learning_rate,
                         period_length=period_length, num_labels=num_labels)

    def training_step(self, batch: Dict, batch_idx: int):
        if self.modeltype == "TS_CXR":
            loss = self(
                x_ts=batch["ts"],  # type ignore
                x_ts_mask=batch["ts_mask"],
                ts_tt_list=batch["ts_tt"],
                reg_ts=batch["reg_ts"],
                cxr_imgs=batch["cxr_imgs"],
                cxr_time=batch["cxr_time"],
                cxr_time_mask=batch["cxr_time_mask"],
                reg_imgs=batch["reg_imgs"],
                reg_imgs_mask=batch["reg_imgs_mask"],
                labels=batch["label"],
            )
        elif self.modeltype == "TS":
            loss = self(
                x_ts=batch["ts"],  # type ignore
                x_ts_mask=batch["ts_mask"],
                ts_tt_list=batch["ts_tt"],
                reg_ts=batch["reg_ts"],
                labels=batch["label"],
            )
        elif self.modeltype == "CXR":
            loss = self(
                cxr_imgs=batch["cxr_imgs"],
                cxr_time=batch["cxr_time"],
                cxr_time_mask=batch["cxr_time_mask"],
                labels=batch["label"],
            )
        else:
            raise NotImplementedError

        batch_size = batch["ts"].size(0)
        if isinstance(loss, Dict):
            self.log_dict({f"train_{k}": v for k, v in loss.items()},
                            on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
            return loss["total_loss"]
        elif isinstance(loss, torch.Tensor):
            self.log("train_loss", loss, on_step=True, on_epoch=True,
                    sync_dist=True, prog_bar=True, batch_size=batch_size)
            return loss
        else:
            raise NotImplementedError

    def on_validation_epoch_start(self) -> None:
        self.validation_step_outputs = []

    def validation_step(self, batch: Dict, batch_idx: int) -> STEP_OUTPUT:
        if self.modeltype == "TS_CXR":
            logits = self(
                x_ts=batch["ts"],  # type ignore
                x_ts_mask=batch["ts_mask"],
                ts_tt_list=batch["ts_tt"],
                reg_ts=batch["reg_ts"],
                cxr_imgs=batch["cxr_imgs"],
                cxr_time=batch["cxr_time"],
                cxr_time_mask=batch["cxr_time_mask"],
                reg_imgs=batch["reg_imgs"],
                reg_imgs_mask=batch["reg_imgs_mask"],
            )
        elif self.modeltype == "TS":
            logits = self(
                x_ts=batch["ts"],  # type ignore
                x_ts_mask=batch["ts_mask"],
                ts_tt_list=batch["ts_tt"],
                reg_ts=batch["reg_ts"],
            )
        elif self.modeltype == "CXR":
            logits = self(
                cxr_imgs=batch["cxr_imgs"],
                cxr_time=batch["cxr_time"],
                cxr_time_mask=batch["cxr_time_mask"],
            )
        else:
            raise NotImplementedError

        return_dict = {
            "logits": logits.detach().cpu().numpy(),
            "label": batch["label"].detach().cpu().numpy()
        }
        self.validation_step_outputs.append(return_dict)

    def on_test_epoch_start(self) -> None:
        self.test_step_outputs = []

    def test_step(self, batch: Dict, batch_idx: int) -> STEP_OUTPUT:
        if self.modeltype == "TS_CXR":
            logits = self(
                x_ts=batch["ts"],  # type ignore
                x_ts_mask=batch["ts_mask"],
                ts_tt_list=batch["ts_tt"],
                reg_ts=batch["reg_ts"],
                cxr_imgs=batch["cxr_imgs"],
                cxr_time=batch["cxr_time"],
                cxr_time_mask=batch["cxr_time_mask"],
                reg_imgs=batch["reg_imgs"],
                reg_imgs_mask=batch["reg_imgs_mask"],
            )
        elif self.modeltype == "TS":
            logits = self(
                x_ts=batch["ts"],  # type ignore
                x_ts_mask=batch["ts_mask"],
                ts_tt_list=batch["ts_tt"],
                reg_ts=batch["reg_ts"],
            )
        elif self.modeltype == "CXR":
            logits = self(
                cxr_imgs=batch["cxr_imgs"],
                cxr_time=batch["cxr_time"],
                cxr_time_mask=batch["cxr_time_mask"],
            )
        else:
            raise NotImplementedError

        return_dict = {
            "logits": logits.detach().cpu().numpy(),
            "label": batch["label"].detach().cpu().numpy()
        }
        self.test_step_outputs.append(return_dict)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {'params': [p for n, p in self.named_parameters()
                        if 'img_encoder' not in n]},
            {'params': [p for n, p in self.named_parameters(
            ) if 'img_encoder' in n], 'lr': self.ts_learning_rate / 5}
        ], lr=self.ts_learning_rate)

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.4, patience=3, verbose=True, mode='max')
        scheduler = {
            'scheduler': lr_scheduler,
            'monitor': 'val_auroc',
            'interval': 'epoch',
            'frequency': 1
        }
        return [optimizer], [scheduler]
