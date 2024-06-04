import argparse
from datetime import datetime
from sklearn.svm import LinearSVC
import sklearn.metrics as metrics
import torch
from torch.utils.data import DataLoader
from lightning import seed_everything, Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, EarlyStopping
from cmehr.utils.file_utils import load_pkl
from cmehr.models.common.linear_finetuner import LinearFinetuner
from cmehr.paths import *
import ipdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True  # type: ignore
torch.backends.cudnn.benchmark = True  # type: ignore
torch.set_float32_matmul_precision("high")

'''
CUDA_VISIBLE_DEVICES=1 python eval_proto_mimic4.py
'''
parser = argparse.ArgumentParser(description="Evaluate MIMIC IV")
parser.add_argument("--eval_method", type=str, default="linear",
                    choices=["svm", "linear", "mlp"])
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--proto_emb_file", type=str, 
                    default="/home/fywang/Documents/EHR_codebase/MMMSPG/prototype_results/mimic4_ihm/ts_proto_embs.pkl")
args = parser.parse_args()


def eval_svm(train_X, train_y, test_X, test_y):
    clf = LinearSVC(dual="auto")
    clf.fit(train_X, train_y)
    y_score = clf.decision_function(test_X)
    auroc = metrics.roc_auc_score(test_y, y_score)
    auprc = metrics.average_precision_score(test_y, y_score)
    f1 = metrics.f1_score(test_y, y_score > 0)
    print(f"[TS]\tAUROC: {auroc}, AUPRC: {auprc}, F1: {f1}")


def eval_linear(train_X, train_y, val_X, val_y, test_X, test_y):
    train_loader = DataLoader(list(zip(train_X, train_y)), batch_size=args.batch_size, 
                              num_workers=args.num_workers, shuffle=True)
    val_loader = DataLoader(list(zip(val_X, val_y)), batch_size=args.batch_size, 
                            num_workers=args.num_workers, shuffle=False)
    test_loader = DataLoader(list(zip(test_X, test_y)), batch_size=args.batch_size, 
                             num_workers=args.num_workers, shuffle=False)

    model = LinearFinetuner(in_size=train_X.shape[1], num_classes=2, 
                            model_type=args.eval_method)
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"mimic4_{args.task}_{args.eval_method}_{run_name}"
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


def cli_main():
    seed_everything(42)
    args.task = "ihm"
    data_dict = load_pkl(args.proto_emb_file)
    train_X = data_dict["train_X"]
    train_y = data_dict["train_Y"]
    val_X = data_dict["val_X"]
    val_y = data_dict["val_Y"]
    test_X = data_dict["test_X"]
    test_y = data_dict["test_Y"]

    if args.eval_method == "svm":
        eval_svm(train_X, train_y, test_X, test_y)
    elif args.eval_method in ["linear", "mlp"]:
        eval_linear(train_X, train_y, val_X, val_y, test_X, test_y)


if __name__ == "__main__":
    cli_main()