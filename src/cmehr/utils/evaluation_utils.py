'''
Utility functions used for evaluation.
'''
from sklearn.svm import LinearSVC
from datetime import datetime
from sklearn.svm import LinearSVC
import sklearn.metrics as metrics
from torch.utils.data import DataLoader
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, EarlyStopping
from cmehr.models.common.linear_finetuner import LinearFinetuner
from cmehr.paths import *


def eval_svm(train_X, train_y, test_X, test_y):
    clf = LinearSVC(dual="auto")
    clf.fit(train_X, train_y)
    y_score = clf.decision_function(test_X)
    auroc = metrics.roc_auc_score(test_y, y_score)
    auprc = metrics.average_precision_score(test_y, y_score)
    f1 = metrics.f1_score(test_y, y_score > 0)
    print(f"AUROC: {auroc}, AUPRC: {auprc}, F1: {f1}")


def eval_linear(train_X, train_y, val_X, val_y, test_X, test_y,
                batch_size=128, task="ihm", eval_method="linear"):
    train_loader = DataLoader(list(zip(train_X, train_y)), batch_size=batch_size, 
                              num_workers=4, shuffle=True)
    val_loader = DataLoader(list(zip(val_X, val_y)), batch_size=batch_size, 
                            num_workers=4, shuffle=False)
    test_loader = DataLoader(list(zip(test_X, test_y)), batch_size=batch_size, 
                             num_workers=4, shuffle=False)

    model = LinearFinetuner(in_size=train_X.shape[1], num_classes=2, 
                            model_type=eval_method)
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"mimic4_{task}_{eval_method}_{run_name}"
    logger = WandbLogger(
        name=run_name,
        save_dir=str(ROOT_PATH / "log"),
        project="cm-ehr", log_model=False)
    callbacks = [LearningRateMonitor(logging_interval="step"), 
                 EarlyStopping(monitor="val_auroc", mode="max", patience=10, verbose=True, min_delta=0.0)]
    trainer = Trainer(max_epochs=100, devices=1, callbacks=callbacks, logger=logger,
                      accelerator="gpu")
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader, ckpt_path="best")