import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score
import math
from argparse import ArgumentParser
from typing import Dict
from lightning import LightningModule
import ipdb
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor, nn
import torch
from torch.optim.optimizer import Optimizer
# from .UTDE_cxr_model import MULTEHRCXRModel


class UTDELightningModule(LightningModule):
    def __init__(self,
                 task: str = "ihm",
                 modeltype: str = "TS_Text",
                 bertcount: int = 3,
                 max_epochs: int = 10,
                 update_bert_epochs: int = 2,
                 txt_learning_rate: float = 2e-5,
                 ts_learning_rate: float = 4e-4,
                 *args,
                 **kwargs
                 ):
        super().__init__()
        self.save_hyperparameters()

        if task == "ihm":
            num_classes = 2
        elif task == "pheno":
            num_classes = 25
        else:
            raise NotImplementedError

        self.model = ProtoMULT(
            task=task,
            num_labels=num_classes
        )
        self.modeltype = modeltype
        self.task = task
        self.max_epochs = max_epochs
        self.bertcount = bertcount
        self.update_bert_epochs = update_bert_epochs
        self.txt_learning_rate = txt_learning_rate
        self.ts_learning_rate = ts_learning_rate

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def on_train_epoch_start(self) -> None:
        if "Text" in self.modeltype:
            count = 0
            if (self.current_epoch % self.update_bert_epochs == 0) and (count < self.bertcount):
                count += 1
                print(f"update bert at epoch: {self.current_epoch}")
                for param in self.model.bertrep.parameters():  # type ignore
                    param.requires_grad = True
            else:
                for param in self.model.bertrep.parameters():  # type ignore
                    param.requires_grad = False

    def training_step(self, batch: Dict, batch_idx: int):
        if batch is None:
            return torch.tensor(0.0).cuda()

        if self.modeltype == "TS_Text":
            loss = self.model(
                x_ts=batch["ts"],  # type ignore
                x_ts_mask=batch["ts_mask"],
                ts_tt_list=batch["ts_tt"],
                reg_ts=batch["reg_ts"],
                input_ids_sequences=batch["input_ids"],
                attn_mask_sequences=batch["attention_mask"],
                note_time_list=batch["note_time"],
                note_time_mask_list=batch["note_time_mask"],
                labels=batch["label"],
            )
        elif self.modeltype == "TS":
            loss = self.model(
                x_ts=batch["ts"],  # type ignore
                x_ts_mask=batch["ts_mask"],
                ts_tt_list=batch["ts_tt"],
                reg_ts=batch["reg_ts"],
                labels=batch["label"]
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
        if batch is None:
            return

        if self.modeltype == "TS_Text":
            logits = self.model(
                x_ts=batch["ts"],  # type ignore
                x_ts_mask=batch["ts_mask"],
                ts_tt_list=batch["ts_tt"],
                reg_ts=batch["reg_ts"],
                input_ids_sequences=batch["input_ids"],
                attn_mask_sequences=batch["attention_mask"],
                note_time_list=batch["note_time"],
                note_time_mask_list=batch["note_time_mask"]
            )
        elif self.modeltype == "TS":
            logits = self.model(
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
        self.validation_step_outputs.append(return_dict)

    def on_shared_epoch_end(self, step_outputs):
        all_logits, all_labels = [], []
        for output in step_outputs:
            all_logits.append(output["logits"])
            all_labels.append(output["label"])
        all_logits = np.concatenate(all_logits, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_pred = np.where(all_logits > 0.5, 1, 0)

        if self.task == "ihm":
            if np.unique(all_labels).shape[0] == 2:
                auroc = roc_auc_score(
                    all_labels, all_logits)
                (precisions, recalls, thresholds) = precision_recall_curve(
                    all_labels, all_logits)
                auprc = auc(recalls, precisions)
                f1 = f1_score(all_labels, all_pred)
                metrics_dict = {
                    "auroc": auroc,
                    "auprc": auprc,
                    "f1": f1
                }
            else:
                # if there is only one class in the labels, then the auroc and auprc are 0
                metrics_dict = {
                    "auroc": 0,
                    "auprc": 0,
                    "f1": 0
                }
        elif self.task == "pheno":
            temp = np.sort(all_labels, axis=0)
            nunique_per_class = (temp[:, 1:] != temp[:, :-1]).sum(axis=0) + 1
            if np.all(nunique_per_class > 1):
                # auc_scores = metrics.roc_auc_score(
                #     all_labels, all_logits, average=None)
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
                metrics_dict['macro_f1'] = f1_score(  # type: ignore
                    all_labels, all_pred, average='macro')
            else:
                # if there is only one class in the labels, then the auroc and auprc are 0
                metrics_dict = {
                    "auroc": 0,
                    "auprc": 0,
                    "f1": 0
                }
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

    def on_test_epoch_start(self) -> None:
        self.test_step_outputs = []

    def test_step(self, batch: Dict, batch_idx: int) -> STEP_OUTPUT:
        if batch is None:
            return

        if self.modeltype == "TS_Text":
            logits = self.model(
                x_ts=batch["ts"],  # type ignore
                x_ts_mask=batch["ts_mask"],
                ts_tt_list=batch["ts_tt"],
                reg_ts=batch["reg_ts"],
                input_ids_sequences=batch["input_ids"],
                attn_mask_sequences=batch["attention_mask"],
                note_time_list=batch["note_time"],
                note_time_mask_list=batch["note_time_mask"]
            )
        elif self.modeltype == "TS":
            logits = self.model(
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

    def on_test_epoch_end(self) -> None:
        metrics_dict = self.on_shared_epoch_end(self.test_step_outputs)
        new_metrics_dict = dict()
        for k, v in metrics_dict.items():
            new_metrics_dict[f"test_{k}"] = v
        self.log_dict(new_metrics_dict, on_epoch=True, sync_dist=True,
                      batch_size=len(self.test_step_outputs))

    def configure_optimizers(self) -> Optimizer:
        if self.modeltype == "TS_Text":
            optimizer = torch.optim.Adam([
                {'params': [p for n, p in self.named_parameters()
                            if 'bert' not in n]},
                {'params': [p for n, p in self.named_parameters(
                ) if 'bert' in n], 'lr': self.txt_learning_rate}
            ], lr=self.ts_learning_rate)
        else:
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.ts_learning_rate)
        return optimizer

    @staticmethod
    def add_specific_args(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument('--txt_learning_rate', type=float, default=2e-5)
        parser.add_argument('--ts_learning_rate', type=float, default=4e-4)
        return parser
