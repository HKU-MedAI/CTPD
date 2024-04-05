from typing import Dict
import ipdb
import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from lightning import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT


class MIMIC4LightningModule(LightningModule):
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
        super().__init__()
        self.save_hyperparameters()

        if task == "ihm":
            num_labels = 2
            period_length = 48
        elif task == "pheno":
            num_labels = 25
            period_length = 24
        else:
            raise NotImplementedError

        self.modeltype = modeltype
        self.task = task
        self.max_epochs = max_epochs
        self.img_learning_rate = img_learning_rate
        self.ts_learning_rate = ts_learning_rate
        self.task = task
        self.tt_max = period_length
        self.num_labels = num_labels

        if self.task == 'ihm':
            self.loss_fct1 = nn.CrossEntropyLoss()
        elif self.task == 'pheno':
            self.loss_fct1 = nn.BCEWithLogitsLoss()
        else:
            raise ValueError("Unknown task")

    def training_step(self, batch: Dict, batch_idx: int):
        if self.modeltype == "TS_CXR":
            loss = self(
                x_ts=batch["ts"],  # type ignore
                x_ts_mask=batch["ts_mask"],
                ts_tt_list=batch["ts_tt"],
                reg_ts=batch["reg_ts"],
                cxr_imgs_sequences=batch["cxr_imgs"],
                cxr_time_sequences=batch["cxr_time"],
                cxr_time_mask_sequences=batch["cxr_time_mask"],
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
                cxr_imgs_sequences=batch["cxr_imgs"],
                cxr_time_sequences=batch["cxr_time"],
                cxr_time_mask_sequences=batch["cxr_time_mask"],
                labels=batch["label"],
            )
        else:
            raise NotImplementedError

        batch_size = batch["ts"].size(0)
        self.log("train_loss", loss, on_step=True, on_epoch=True,
                 sync_dist=True, prog_bar=True, batch_size=batch_size)

        return loss

    def on_validation_epoch_start(self) -> None:
        self.validation_step_outputs = []

    def validation_step(self, batch: Dict, batch_idx: int) -> STEP_OUTPUT:
        if self.modeltype == "TS_CXR":
            logits = self(
                x_ts=batch["ts"],  # type ignore
                x_ts_mask=batch["ts_mask"],
                ts_tt_list=batch["ts_tt"],
                reg_ts=batch["reg_ts"],
                cxr_imgs_sequences=batch["cxr_imgs"],
                cxr_time_sequences=batch["cxr_time"],
                cxr_time_mask_sequences=batch["cxr_time_mask"],
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
                cxr_imgs_sequences=batch["cxr_imgs"],
                cxr_time_sequences=batch["cxr_time"],
                cxr_time_mask_sequences=batch["cxr_time_mask"],
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
                cxr_imgs_sequences=batch["cxr_imgs"],
                cxr_time_sequences=batch["cxr_time"],
                cxr_time_mask_sequences=batch["cxr_time_mask"],
            )
        elif self.modeltype == "TS":
            logits = self(
                x_ts=batch["ts"],  # type ignore
                x_ts_mask=batch["ts_mask"],
                ts_tt_list=batch["ts_tt"],
                reg_ts=batch["reg_ts"],
            )
        else:
            raise NotImplementedError

        return_dict = {
            "logits": logits.detach().cpu().numpy(),
            "label": batch["label"].detach().cpu().numpy()
        }
        self.test_step_outputs.append(return_dict)

    def on_shared_epoch_end(self, step_outputs):
        all_logits, all_labels = [], []
        for output in step_outputs:
            all_logits.append(output["logits"])
            all_labels.append(output["label"])
        all_logits = np.concatenate(all_logits, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        if self.task == "ihm":
            if np.unique(all_labels).shape[0] == 2:
                auroc = metrics.roc_auc_score(
                    all_labels, all_logits)
                auprc = metrics.average_precision_score(
                    all_labels, all_logits)
                metrics_dict = {
                    "auroc": auroc,
                    "auprc": auprc,
                }
                for thres in [0.2, 0.3, 0.5, 0.7, 0.9]:
                    all_pred = np.where(all_logits > thres, 1, 0)
                    f1 = metrics.f1_score(all_labels, all_pred)
                    metrics_dict[f"f1_{thres}"] = f1
            else:
                # if there is only one class in the labels, then the auroc and auprc are 0
                metrics_dict = {
                    "auroc": 0,
                    "auprc": 0,
                    "f1": 0
                }
            self.report_auroc = metrics_dict["auroc"]
            self.report_auprc = metrics_dict["auprc"]
            self.report_f1_2 = metrics_dict["f1_0.2"]
            self.report_f1_3 = metrics_dict["f1_0.3"]
            self.report_f1_5 = metrics_dict["f1_0.5"]
            self.report_f1_7 = metrics_dict["f1_0.7"]
            self.report_f1_9 = metrics_dict["f1_0.9"]

        elif self.task == "pheno":
            temp = np.sort(all_labels, axis=0)
            nunique_per_class = (temp[:, 1:] != temp[:, :-1]).sum(axis=0) + 1
            if np.all(nunique_per_class > 1):
                ave_auc_micro = metrics.roc_auc_score(all_labels, all_logits,
                                                      average="micro")
                ave_auc_macro = metrics.roc_auc_score(all_labels, all_logits,
                                                      average="macro")
                ave_auc_weighted = metrics.roc_auc_score(all_labels, all_logits,
                                                         average="weighted")

                metrics_dict = {  # "auc_scores": auc_scores,
                    "auroc_micro": ave_auc_micro,
                    "auroc_macro": ave_auc_macro,
                    "auroc_weighted": ave_auc_weighted}

                for thres in [0.2, 0.3, 0.5, 0.7, 0.9]:
                    all_pred = np.where(all_logits > thres, 1, 0)
                    f1 = metrics.f1_score(
                        all_labels, all_pred, average='macro')
                    metrics_dict[f"macro_f1_{thres}"] = f1

                auprc = metrics.average_precision_score(
                    all_labels, all_logits, average="macro")
                metrics_dict["auprc"] = auprc
            else:
                # if there is only one class in the labels, then the auroc and auprc are 0
                metrics_dict = {
                    "auprc": 0,
                    "auroc_micro": 0,
                    "auroc_macro": 0,
                    "auroc_weighted": 0
                }
                for thres in [0.2, 0.3, 0.5, 0.7, 0.9]:
                    metrics_dict[f"macro_f1_{thres}"] = 0

            self.report_auroc = metrics_dict["auroc_macro"]
            self.report_auprc = metrics_dict["auprc"]
            self.report_f1_2 = metrics_dict["macro_f1_0.2"]
            self.report_f1_3 = metrics_dict["macro_f1_0.3"]
            self.report_f1_5 = metrics_dict["macro_f1_0.5"]
            self.report_f1_7 = metrics_dict["macro_f1_0.7"]
            self.report_f1_9 = metrics_dict["macro_f1_0.9"]

        else:
            raise NotImplementedError

        return metrics_dict

    def on_validation_epoch_end(self) -> None:
        metrics_dict = self.on_shared_epoch_end(self.validation_step_outputs)
        new_metrics_dict = dict()
        for k, v in metrics_dict.items():
            new_metrics_dict[f"val_{k}"] = v
        self.log_dict(new_metrics_dict, on_epoch=True, sync_dist=True,
                      batch_size=len(self.validation_step_outputs))

    def on_test_epoch_end(self) -> None:
        metrics_dict = self.on_shared_epoch_end(self.test_step_outputs)
        new_metrics_dict = dict()
        for k, v in metrics_dict.items():
            new_metrics_dict[f"test_{k}"] = v
        self.log_dict(new_metrics_dict, on_epoch=True, sync_dist=True,
                      batch_size=len(self.test_step_outputs))

    def configure_optimizers(self):
        if self.modeltype == "TS_CXR":
            optimizer = torch.optim.Adam([
                {'params': [p for n, p in self.named_parameters()
                            if 'img_encoder' not in n]},
                {'params': [p for n, p in self.named_parameters(
                ) if 'img_encoder' in n], 'lr': self.img_learning_rate}
            ], lr=self.ts_learning_rate)
            return optimizer
        else:
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.ts_learning_rate)
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=0.4, patience=3, verbose=True, mode='max')
            scheduler = {
                'scheduler': lr_scheduler,
                'monitor': 'val_auprc' if self.task == 'ihm' else 'val_auroc_macro',
                'interval': 'epoch',
                'frequency': 1
            }
            return [optimizer], [scheduler]
