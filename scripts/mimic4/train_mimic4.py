'''
This is the script to run experiments on MIMIC-IV dataset.
'''

from argparse import ArgumentParser
from datetime import datetime
import pandas as pd
import ipdb

import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from cmehr.dataset.mimic4_downstream_datamodule import MIMIC4DataModule
from cmehr.models.mimic4 import (
    CNNModule, ProtoTSModel, IPNetModule, GRUDModule, SEFTModule, RNNModule, LSTMModule,
    MTANDModule, DGM2OModule, MedFuseModule, TransformerModule, MILLETModule, OTKModule,
    CAMELOTModule, TSLANETModule, CTPDModule)
from cmehr.paths import *


torch.backends.cudnn.deterministic = True  # type: ignore
torch.backends.cudnn.benchmark = True  # type: ignore
torch.set_float32_matmul_precision("high")


'''
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_mimic4.py --task ihm --model_name CTPD --devices 4 --batch_size 12 \

CUDA_VISIBLE_DEVICES=0,1,2,3 python train_mimic4.py --task pheno --model_name CTPD --devices 4 --batch_size 12 \
    --use_prototype --use_multiscale
'''
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
parser.add_argument("--model_name", type=str, default="cnn",
                    choices=["proto_ts", "ipnet", "grud", "seft", "mtand", "dgm2", "rnn",
                             "medfuse", "cnn", "lstm", "transformer", "millet", "camelot",
                             "otk", "diffem", "tslanet", "CTPD"])
parser.add_argument("--modeltype", type=str, default="TS_CXR",
                    choices=["TS_CXR", "TS", "CXR"],
                    help="Set the model type to use for training")
parser.add_argument("--ts_learning_rate", type=float, default=4e-5)
parser.add_argument("--ckpt_path", type=str,
                    default="")
parser.add_argument("--test_only", action="store_true")
parser.add_argument("--pooling_type", type=str, default="mean",
                    choices=["attention", "mean", "last"])
parser.add_argument("--use_prototype", action="store_true")
parser.add_argument("--use_multiscale", action="store_true")
parser.add_argument("--lamb1", type=float, default=1.)
parser.add_argument("--lamb2", type=float, default=1.)
parser.add_argument("--lamb3", type=float, default=0)
parser.add_argument("--num_slots", type=int, default=16)
args = parser.parse_args()


def cli_main():

    all_auroc = []
    all_auprc = []
    all_f1 = []

    for seed in [41, 42, 43]:
        seed_everything(seed)

        # This is fixed for MIMIC4
        args.orig_d_ts = 15
        args.orig_reg_d_ts = 30

        # define datamodule
        if args.first_nrows == -1:
            args.first_nrows = None

        if args.task == "ihm":
            args.period_length = 48
        elif args.task == "pheno":
            args.period_length = 24

        dm = MIMIC4DataModule(
            mimic_cxr_dir=str(MIMIC_CXR_JPG_PATH),
            # by default, we use this multimodal dataset.
            file_path=str(
                DATA_PATH / f"output_mimic4/TS_CXR/{args.task}"),
            modeltype=args.modeltype,
            period_length=args.period_length,
            batch_size=args.batch_size,
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
        elif args.model_name == "rnn":
            if args.ckpt_path:
                model = RNNModule.load_from_checkpoint(
                    args.ckpt_path, **vars(args))
            else:
                model = RNNModule(**vars(args))
        elif args.model_name == "lstm":
            if args.ckpt_path:
                model = LSTMModule.load_from_checkpoint(
                    args.ckpt_path, **vars(args))
            else:
                model = LSTMModule(**vars(args))
        elif args.model_name == "cnn":
            if args.ckpt_path:
                model = CNNModule.load_from_checkpoint(
                    args.ckpt_path, **vars(args))
            else:
                model = CNNModule(**vars(args))
        elif args.model_name == "transformer":
            if args.ckpt_path:
                model = TransformerModule.load_from_checkpoint(
                    args.ckpt_path, **vars(args))
            else:
                model = TransformerModule(**vars(args))
        elif args.model_name == "millet":
            if args.ckpt_path:
                model = MILLETModule.load_from_checkpoint(
                    args.ckpt_path, **vars(args))
            else:
                model = MILLETModule(**vars(args))
        elif args.model_name == "otk":
            if args.ckpt_path:
                model = OTKModule.load_from_checkpoint(
                    args.ckpt_path, **vars(args))
            else:
                model = OTKModule(**vars(args))
        elif args.model_name == "camelot":
            if args.ckpt_path:
                model = CAMELOTModule.load_from_checkpoint(
                    args.ckpt_path, **vars(args))
            else:
                model = CAMELOTModule(**vars(args))
        elif args.model_name == "tslanet":
            if args.ckpt_path:
                model = TSLANETModule.load_from_checkpoint(
                    args.ckpt_path, **vars(args))
            else:
                model = TSLANETModule(**vars(args))
        elif args.model_name == "CTPD":
            if args.ckpt_path:
                model = CTPDModule.load_from_checkpoint(
                    args.ckpt_path, **vars(args))
            else:
                model = CTPDModule(**vars(args))
        else:
            raise ValueError("Invalid model name")

        model.train_iters_per_epoch = len(dm.train_dataloader()) // (args.accumulate_grad_batches * args.devices)

        # initialize trainer
        run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"mimic4_{args.task}_{args.model_name}_{run_name}"
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
                    monitor="val_auroc",
                    mode="max",
                    save_top_k=2,
                    save_last=False),
                EarlyStopping(monitor="val_auroc", patience=10,
                              mode="max", verbose=True)
            ]
        elif args.task == "pheno":
            callbacks = [
                LearningRateMonitor(logging_interval="step"),
                ModelCheckpoint(
                    dirpath=str(ROOT_PATH / "log/ckpts" / run_name),
                    monitor="val_auroc",
                    mode="max",
                    save_top_k=2,
                    save_last=False),
                EarlyStopping(monitor="val_auroc", patience=10,
                              mode="max", verbose=True)
            ]
        trainer = Trainer(
            devices=args.devices,
            accelerator="gpu",
            max_epochs=args.max_epochs,
            # precision="16-mixed",
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

        # close wandb
        import wandb
        wandb.finish()

        all_auroc.append(model.report_auroc)
        all_auprc.append(model.report_auprc)
        all_f1.append(model.report_f1)

    report_df = pd.DataFrame({
        "auroc": all_auroc,
        "auprc": all_auprc,
        "f1": all_f1
    })

    mean_df = report_df.mean(axis=0) * 100
    std_df = report_df.std(axis=0) * 100
    statistic_df = pd.concat([mean_df, std_df], axis=1)
    statistic_df.columns = ["mean", "std"]
    print(statistic_df)


if __name__ == "__main__":
    cli_main()
