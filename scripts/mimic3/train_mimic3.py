'''
This is the script to run TS experiments on MIMIC-III dataset.
'''

from argparse import ArgumentParser
from datetime import datetime
import pandas as pd
import ipdb

import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from cmehr.dataset import MIMIC4DataModule, MIMIC3DataModule
from cmehr.models.mimic4 import (
    CNNModule, ProtoTSModel, IPNetModule, GRUDModule, SEFTModule,
    MTANDModule, DGM2OModule, MedFuseModule)
from cmehr.paths import *

torch.backends.cudnn.deterministic = True  # type: ignore
torch.backends.cudnn.benchmark = True  # type: ignore
torch.set_float32_matmul_precision("high")

parser = ArgumentParser(description="PyTorch Lightning EHR Model")
parser.add_argument("--task", type=str, default="pheno",
                    choices=["ihm", "decomp", "los", "pheno"])
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--update_counts", type=int, default=3)
parser.add_argument("--max_epochs", type=int, default=50)
parser.add_argument("--update_encoder_epochs", type=int, default=2)
parser.add_argument("--devices", type=int, default=1)
parser.add_argument("--max_length", type=int, default=1024)
parser.add_argument("--accumulate_grad_batches", type=int, default=1)
parser.add_argument("--first_nrows", type=int, default=-1)
parser.add_argument("--model_name", type=str, default="medfuse",
                    choices=["proto_ts", "ipnet", "grud", "seft", "mtand", "dgm2",
                             "medfuse", "cnn"])
parser.add_argument("--ts_learning_rate", type=float, default=4e-4)
parser.add_argument("--ckpt_path", type=str,
                    default="")
parser.add_argument("--test_only", action="store_true")
parser.add_argument("--pooling_type", type=str, default="mean",
                    choices=["attention", "mean", "last"])
parser.add_argument("--use_prototype", action="store_true")
parser.add_argument("--use_multiscale", action="store_true")
args = parser.parse_args()

'''
CUDA_VISIBLE_DEVICES=2 python train_mimic3.py --devices 1 --task ihm --batch_size 128 --model_name cnn
'''

args.orig_reg_d_ts = 34
args.orig_d_ts = 17


def cli_main():
    all_auroc = []
    all_auprc = []
    all_f1_2 = []
    all_f1_3 = []
    all_f1_5 = []
    all_f1_7 = []
    all_f1_9 = []

    for seed in [41, 42, 43]:
        seed_everything(seed)

        # define datamodule
        if args.first_nrows == -1:
            args.first_nrows = None

        if args.task == "ihm":
            args.period_length = 48
        elif args.task == "pheno":
            args.period_length = 24

        dm = MIMIC3DataModule(
            file_path=str(
                DATA_PATH / f"output_mimic3/{args.task}"),
            tt_max=args.period_length,
            batch_size=args.batch_size,
            modeltype="TS",
            num_workers=args.num_workers,
            first_nrows=args.first_nrows)

        # define model
        if args.test_only:
            args.devices = 1

        if args.model_name == "ipnet":
            if args.ckpt_path:
                model = IPNetModule.load_from_checkpoint(
                    args.ckpt_path, **vars(args))
            else:
                model = IPNetModule(**vars(args))
        elif args.model_name == "proto_ts":
            if args.ckpt_path:
                model = ProtoTSModel.load_from_checkpoint(
                    args.ckpt_path, **vars(args))
            else:
                model = ProtoTSModel(**vars(args))
        elif args.model_name == "grud":
            if args.ckpt_path:
                model = GRUDModule.load_from_checkpoint(
                    args.ckpt_path, **vars(args))
            else:
                model = GRUDModule(**vars(args))
        elif args.model_name == "seft":
            if args.ckpt_path:
                model = SEFTModule.load_from_checkpoint(
                    args.ckpt_path, **vars(args))
            else:
                model = SEFTModule(**vars(args))
        elif args.model_name == "mtand":
            if args.ckpt_path:
                model = MTANDModule.load_from_checkpoint(
                    args.ckpt_path, **vars(args))
            else:
                model = MTANDModule(**vars(args))
        elif args.model_name == "dgm2":
            if args.ckpt_path:
                model = DGM2OModule.load_from_checkpoint(
                    args.ckpt_path, **vars(args))
            else:
                model = DGM2OModule(**vars(args))
        elif args.model_name == "medfuse":
            if args.ckpt_path:
                model = MedFuseModule.load_from_checkpoint(
                    args.ckpt_path, **vars(args))
            else:
                model = MedFuseModule(**vars(args))
        elif args.model_name == "cnn":
            if args.ckpt_path:
                model = CNNModule.load_from_checkpoint(
                    args.ckpt_path, **vars(args))
            else:
                model = CNNModule(**vars(args))
        else:
            raise ValueError("Invalid model name")

        # initialize trainer
        run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"mimic_{args.task}_{args.model_name}_{run_name}"
        os.makedirs(ROOT_PATH / "log/ckpts", exist_ok=True)
        logger = WandbLogger(
            name=run_name,
            save_dir=str(ROOT_PATH / "log"),
            project="cm-ehr", log_model=False)
        if args.task == "ihm":
            callbacks = [
                LearningRateMonitor(logging_interval="step"),
                ModelCheckpoint(
                    dirpath=str(ROOT_PATH / "log/ckpts" / run_name),
                    monitor="val_auprc",
                    mode="max",
                    save_top_k=2,
                    save_last=False),
                EarlyStopping(monitor="val_auprc", patience=5,
                              mode="max", verbose=True)
            ]
        elif args.task == "pheno":
            callbacks = [
                LearningRateMonitor(logging_interval="step"),
                ModelCheckpoint(
                    dirpath=str(ROOT_PATH / "log/ckpts" / run_name),
                    monitor="val_auroc_macro",
                    mode="max",
                    save_top_k=2,
                    save_last=False),
                EarlyStopping(monitor="val_auroc_macro", patience=5,
                              mode="max", verbose=True)
            ]
        trainer = Trainer(
            devices=args.devices,
            accelerator="gpu",
            max_epochs=args.max_epochs,
            precision="16-mixed",
            accumulate_grad_batches=args.accumulate_grad_batches,
            # deterministic=False,
            callbacks=callbacks,
            logger=logger,
            strategy="ddp_find_unused_parameters_true",
        )

        if not args.test_only:
            trainer.fit(model, dm)
            trainer.test(model, datamodule=dm, ckpt_path="best")
        else:
            trainer.test(model, datamodule=dm)

        all_auroc.append(model.report_auroc)
        all_auprc.append(model.report_auprc)
        all_f1_2.append(model.report_f1_2)
        all_f1_3.append(model.report_f1_3)
        all_f1_5.append(model.report_f1_5)
        all_f1_7.append(model.report_f1_7)
        all_f1_9.append(model.report_f1_9)

    report_df = pd.DataFrame({
        "auroc": all_auroc,
        "auprc": all_auprc,
        "f1_2": all_f1_2,
        "f1_3": all_f1_3,
        "f1_5": all_f1_5,
        "f1_7": all_f1_7,
        "f1_9": all_f1_9
    })

    mean_df = report_df.mean(axis=0)
    std_df = report_df.std(axis=0)
    statistic_df = pd.concat([mean_df, std_df], axis=1)
    statistic_df.columns = ["mean", "std"]
    print(statistic_df)


if __name__ == "__main__":
    cli_main()
