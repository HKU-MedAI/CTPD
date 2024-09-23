'''
Utility functions used for evaluation.
'''
import ipdb
import torch
import numpy as np
from torch.utils.data import Dataset
from datetime import datetime
from sklearn.svm import LinearSVC
import sklearn.metrics as metrics
from torch.utils.data import DataLoader
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, EarlyStopping
from cmehr.models.common.linear_finetuner import LinearFinetuner
from cmehr.paths import *


def eval_svm(train_X, train_y, val_X, val_y, test_X, test_y):
    if train_y.ndim == 1:
        clf = LinearSVC(dual="auto")
        clf.fit(train_X, train_y)
        y_score = clf.decision_function(test_X)
        auroc = metrics.roc_auc_score(test_y, y_score)
        auprc = metrics.average_precision_score(test_y, y_score)
        val_y_score = clf.decision_function(val_X)
        # select the best threshold on the validation set
        _, _, thres = metrics.precision_recall_curve(val_y, val_y_score)
        all_val_f1 = []
        for t in thres:
            y_pred = val_y_score > t
            f1 = metrics.f1_score(val_y, y_pred)
            all_val_f1.append(f1)
        best_thres = thres[np.argmax(all_val_f1)]
        f1 = metrics.f1_score(test_y, y_score > best_thres)
        print(f"AUROC: {auroc}, AUPRC: {auprc}, F1: {f1}")
    else:
        num_classes = train_y.shape[1]
        class_auroc = []
        class_auprc = []
        class_f1 = []
        for i in range(num_classes):
            clf = LinearSVC(dual="auto")
            clf.fit(train_X, train_y[:, i])
            y_score = clf.decision_function(test_X)
            auroc = metrics.roc_auc_score(test_y[:, i], y_score)
            auprc = metrics.average_precision_score(test_y[:, i], y_score)
            val_y_score = clf.decision_function(val_X)
            # select the best threshold on the validation set
            _, _, thres = metrics.precision_recall_curve(val_y[:, i], val_y_score)
            all_val_f1 = []
            for t in thres:
                y_pred = val_y_score > t
                f1 = metrics.f1_score(val_y[:, i], y_pred)
                all_val_f1.append(f1)
            best_thres = thres[np.argmax(all_val_f1)]
            f1 = metrics.f1_score(test_y[:, i], y_score > best_thres)
            # print(f"AUROC: {auroc}, AUPRC: {auprc}, F1: {f1}")
            class_auroc.append(auroc)
            class_auprc.append(auprc)
            class_f1.append(f1)
        print(f"Mean AUROC: {np.mean(class_auroc)}, Mean AUPRC: {np.mean(class_auprc)}, Mean F1: {np.mean(class_f1)}")


class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X) 

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def eval_linear(train_X, train_y, val_X, val_y, test_X, test_y,
                n_proto=50, batch_size=128, task="ihm", eval_method="linear"):
    train_loader = DataLoader(CustomDataset(train_X, train_y), batch_size=batch_size, 
                              num_workers=4, shuffle=True)
    val_loader = DataLoader(CustomDataset(val_X, val_y), batch_size=batch_size, 
                            num_workers=4, shuffle=False)
    test_loader = DataLoader(CustomDataset(test_X, test_y), batch_size=batch_size, 
                             num_workers=4, shuffle=False)

    model = LinearFinetuner(in_size=train_X.shape[1], num_classes=2, n_proto=n_proto,
                            model_type=eval_method)
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"mimic4_{task}_{eval_method}_{run_name}"
    logger = WandbLogger(
        name=run_name,
        save_dir=str(ROOT_PATH / "log"),
        project="cm-ehr", log_model=False)
    callbacks = [LearningRateMonitor(logging_interval="step"), 
                 EarlyStopping(monitor="val_auroc", mode="max", patience=10, verbose=True, min_delta=0.0)]
    trainer = Trainer(max_epochs=100, 
                      devices=1, 
                      callbacks=callbacks, 
                      logger=logger,
                      accelerator="gpu",
                      precision="16-mixed")
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader, ckpt_path="best")