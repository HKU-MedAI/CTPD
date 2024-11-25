from typing import Dict
import ipdb
import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn as nn
from lightning import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT


class MIMIC3LightningModule(LightningModule):
    '''
    Base lightning model on MIMIC III dataset.
    '''

    def __init__(self,
                 task: str = "ihm",
                 modeltype: str = "TS_CXR",
                 max_epochs: int = 10,
                 ts_learning_rate: float = 4e-4,
                 period_length: int = 48,
                 *args,
                 **kwargs
                 ):
        super().__init__()
        self.save_hyperparameters()

        if task in ["ihm", "readm"]:
            num_labels = 2
            period_length = 48
            self.best_thres = [0]
        elif task == "pheno":
            num_labels = 25
            period_length = 24
            self.best_thres = [0] * num_labels
        else:
            raise NotImplementedError
        
        self.modeltype = modeltype
        self.task = task
        self.max_epochs = max_epochs
        self.ts_learning_rate = ts_learning_rate
        self.task = task
        self.tt_max = period_length
        self.num_labels = num_labels

        if self.task in ['ihm', 'readm']:
            self.loss_fct1 = nn.CrossEntropyLoss()
        elif self.task == 'pheno':
            self.loss_fct1 = nn.BCEWithLogitsLoss()
        else:
            raise ValueError("Unknown task")

    def training_step(self, batch: Dict, batch_idx: int):
        loss = self(
            x_ts=batch["ts"],  # type ignore
            x_ts_mask=batch["ts_mask"],
            ts_tt_list=batch["ts_tt"],
            reg_ts=batch["reg_ts"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            note_time=batch["note_time"],
            note_time_mask=batch["note_time_mask"],
            labels=batch["label"],
        )
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

        return loss

    def on_validation_epoch_start(self) -> None:
        self.validation_step_outputs = []

    def validation_step(self, batch: Dict, batch_idx: int) -> STEP_OUTPUT:
        logits = self(
            x_ts=batch["ts"],  # type ignore
            x_ts_mask=batch["ts_mask"],
            ts_tt_list=batch["ts_tt"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            note_time=batch["note_time"],
            note_time_mask=batch["note_time_mask"],
            reg_ts=batch["reg_ts"],
        )
        return_dict = {
            "logits": logits.detach().cpu().numpy(),
            "label": batch["label"].detach().cpu().numpy()
        }
        self.validation_step_outputs.append(return_dict)

    def on_test_epoch_start(self) -> None:
        self.test_step_outputs = []

    def test_step(self, batch: Dict, batch_idx: int) -> STEP_OUTPUT:
        logits = self(
            x_ts=batch["ts"],  # type ignore
            x_ts_mask=batch["ts_mask"],
            ts_tt_list=batch["ts_tt"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            note_time=batch["note_time"],
            note_time_mask=batch["note_time_mask"],
            reg_ts=batch["reg_ts"]
        )

        return_dict = {
            "logits": logits.detach().cpu().numpy(),
            "label": batch["label"].detach().cpu().numpy()
        }
        self.test_step_outputs.append(return_dict)

    def on_shared_epoch_end(self, step_outputs, split="val"):
        all_logits, all_labels = [], []
        for output in step_outputs:
            all_logits.append(output["logits"])
            all_labels.append(output["label"])
        all_logits = np.concatenate(all_logits, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        if self.task in ["ihm", "readm"]:
            if np.unique(all_labels).shape[0] == 2:
                auroc = metrics.roc_auc_score(
                    all_labels, all_logits)
                auprc = metrics.average_precision_score(
                    all_labels, all_logits)
                metrics_dict = {
                    "auroc": auroc,
                    "auprc": auprc,
                }
                # select best thresholds in val set
                if split == "val":
                    _, _, thresholds = metrics.roc_curve(all_labels, all_logits)
                    thresholds = thresholds[1:]
                    all_f1_scores = []
                    for thres in thresholds:
                        cur_pred = np.where(all_logits > thres, 1, 0)
                        f1 = metrics.f1_score(all_labels, cur_pred)
                        all_f1_scores.append(f1)
                    best_thres = thresholds[np.argmax(all_f1_scores)]
                    self.best_thres[0] = best_thres
                    metrics_dict["f1"] = np.max(all_f1_scores)
                elif split == "test":
                    best_thres = self.best_thres[0]
                    cur_pred = np.where(all_logits > best_thres, 1, 0)
                    f1 = metrics.f1_score(all_labels, cur_pred)
                    metrics_dict["f1"] = f1
            else:
                # if there is only one class in the labels, then the auroc and auprc are 0
                metrics_dict = {
                    "auroc": 0,
                    "auprc": 0,
                    "f1": 0
                }
            self.report_auroc = metrics_dict["auroc"]
            self.report_auprc = metrics_dict["auprc"]
            self.report_f1 = metrics_dict["f1"]

        elif self.task == "pheno":
            temp = np.sort(all_labels, axis=0)
            nunique_per_class = (temp[:, 1:] != temp[:, :-1]).sum(axis=0) + 1
            if np.all(nunique_per_class > 1):
                ave_auc_macro = metrics.roc_auc_score(all_labels, all_logits,
                                                      average="macro")
                auprc = metrics.average_precision_score(
                    all_labels, all_logits, average="macro")
                metrics_dict = { 
                    "auroc": ave_auc_macro,
                    "auprc": auprc
                    }

                if split == "val":
                    cur_f1 = []
                    for i in range(self.num_labels):
                        _, _, thresholds = metrics.roc_curve(all_labels[:, i], all_logits[:, i])
                        thresholds = thresholds[1:]
                        all_f1_scores = []
                        for thres in thresholds:
                            cur_pred = np.where(all_logits[:, i] > thres, 1, 0)
                            f1 = metrics.f1_score(all_labels[:, i], cur_pred)
                            all_f1_scores.append(f1)
                        best_thres = thresholds[np.argmax(all_f1_scores)]
                        self.best_thres[i] = best_thres
                        cur_f1.append(np.max(all_f1_scores))
                    metrics_dict["f1"] = np.mean(cur_f1)
                else:
                    cur_f1 = []
                    for i in range(self.num_labels):
                        best_thres = self.best_thres[i]
                        cur_pred = np.where(all_logits[:, i] > best_thres, 1, 0)
                        f1 = metrics.f1_score(all_labels[:, i], cur_pred)
                        cur_f1.append(f1)
                    metrics_dict["f1"] = np.mean(cur_f1)
            else:
                # if there is only one class in the labels, then the auroc and auprc are 0
                metrics_dict = {
                    "auprc": 0,
                    "auroc": 0,
                    "f1": 0
                }

            self.report_auroc = metrics_dict["auroc"]
            self.report_auprc = metrics_dict["auprc"]
            self.report_f1 = metrics_dict["f1"]

        else:
            raise NotImplementedError

        return metrics_dict

    def on_validation_epoch_end(self) -> None:
        metrics_dict = self.on_shared_epoch_end(self.validation_step_outputs, split="val")
        new_metrics_dict = dict()
        for k, v in metrics_dict.items():
            new_metrics_dict[f"val_{k}"] = v
        self.log_dict(new_metrics_dict, on_epoch=True, sync_dist=True,
                      batch_size=len(self.validation_step_outputs))

    def on_test_epoch_end(self) -> None:
        metrics_dict = self.on_shared_epoch_end(self.test_step_outputs, split="test")
        new_metrics_dict = dict()
        for k, v in metrics_dict.items():
            new_metrics_dict[f"test_{k}"] = v
        self.log_dict(new_metrics_dict, on_epoch=True, sync_dist=True,
                      batch_size=len(self.test_step_outputs))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
                {'params': [p for n, p in self.named_parameters()
                            if 'bert' not in n]},
                {'params': [p for n, p in self.named_parameters(
                ) if 'bert' in n], 'lr': self.ts_learning_rate / 5}
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
    

class MIMIC3NoteModule(MIMIC3LightningModule):
    ''' Specialized for note data. '''
    def __init__(self, 
                 task: str = "ihm", 
                 modeltype: str = "TS_CXR", 
                 max_epochs: int = 100, 
                 ts_learning_rate: float = 0.0004, 
                 period_length: int = 48, 
                 *args, 
                 **kwargs):
        super().__init__(task, modeltype, max_epochs, ts_learning_rate, period_length, *args, **kwargs)

    def training_step(self, batch: Dict, batch_idx: int):
        if batch is None or batch["input_ids"] is None:
            return torch.tensor(0.0).type_as(batch["input_ids"])
        loss = self(
            input_ids_sequences=batch["input_ids"],
            attn_mask_sequences=batch["attention_mask"],
            note_time_list=batch["note_time"],
            note_time_mask_list=batch["note_time_mask"],
            labels=batch["label"]
        )
        batch_size = batch["input_ids"].size(0)
        self.log("train_loss", loss, on_step=True, on_epoch=True,
                 sync_dist=True, prog_bar=True, batch_size=batch_size)

        return loss

    def on_validation_epoch_start(self) -> None:
        self.validation_step_outputs = []

    def validation_step(self, batch: Dict, batch_idx: int) -> STEP_OUTPUT:
        if batch is None or batch["input_ids"] is None:
            return
        logits = self(
            input_ids_sequences=batch["input_ids"],
            attn_mask_sequences=batch["attention_mask"],
            note_time_list=batch["note_time"],
            note_time_mask_list=batch["note_time_mask"]
        )
        return_dict = {
            "logits": logits.detach().cpu().numpy(),
            "label": batch["label"].detach().cpu().numpy()
        }
        self.validation_step_outputs.append(return_dict)

    def on_test_epoch_start(self) -> None:
        self.test_step_outputs = []

    def test_step(self, batch: Dict, batch_idx: int) -> STEP_OUTPUT:
        if batch is None or batch["input_ids"] is None:
            return
        logits = self(
            input_ids_sequences=batch["input_ids"],
            attn_mask_sequences=batch["attention_mask"],
            note_time_list=batch["note_time"],
            note_time_mask_list=batch["note_time_mask"]
        )

        return_dict = {
            "logits": logits.detach().cpu().numpy(),
            "label": batch["label"].detach().cpu().numpy()
        }
        self.test_step_outputs.append(return_dict)