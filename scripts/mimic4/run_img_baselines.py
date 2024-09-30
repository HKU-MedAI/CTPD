'''
This is the script to run experiments on MIMIC-III dataset.
'''

from argparse import ArgumentParser
from datetime import datetime
import pandas as pd
import ipdb

import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
# from cmehr.dataset import MIMIC3DataModule
from cmehr.dataset import MIMIC4DataModule
from cmehr.models.mimic4.CXR.mtand_model import MIMIC4MTANDModule
# from cmehr.models.mimic4.CXR.grud_model import GRUDModule
from cmehr.models.mimic4.CXR.flat_model import MIMIC4FlatModule
from cmehr.models.mimic4.CXR.transformer_model import MIMIC4HierTransformerModule
from cmehr.models.mimic4.CXR.tlstm_model import MIMIC4TLSTMModule
# from cmehr.models.mimic4.CXR.ftlstm_model import FTLSTMModule
from cmehr.paths import *

torch.backends.cudnn.deterministic = True  # type: ignore
torch.backends.cudnn.benchmark = True  # type: ignore
torch.set_float32_matmul_precision("high")

'''
CUDA_VISIBLE_DEVICES=6 python run_img_baselines.py --task ihm --model_name mtand
CUDA_VISIBLE_DEVICES=7 python run_img_baselines.py --task pheno --model_name mtand
'''
parser = ArgumentParser(description="PyTorch Lightning EHR Model") 
parser.add_argument("--task", type=str, default="pheno",
                    choices=["ihm", "decomp", "los", "pheno"])
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--update_counts", type=int, default=3)
parser.add_argument("--max_epochs", type=int, default=50)
parser.add_argument("--update_encoder_epochs", type=int, default=2)
parser.add_argument("--devices", type=int, default=1)
parser.add_argument("--max_length", type=int, default=1024)
parser.add_argument("--accumulate_grad_batches", type=int, default=1)
parser.add_argument("--first_nrows", type=int, default=-1)
parser.add_argument("--num_imgs", type=int, default=5)
parser.add_argument("--model_name", type=str, default="mtand",
                    choices=["mtand", "grud", "flat", "hiertrans",
                             "tlstm", "ftlstm"])
parser.add_argument("--ts_learning_rate", type=float, default=4e-4)
parser.add_argument("--ckpt_path", type=str,
                    default="")
parser.add_argument("--test_only", action="store_true")
parser.add_argument("--pooling_type", type=str, default="mean",
                    choices=["attention", "mean", "last"])
parser.add_argument("--bert_type", type=str, default="prajjwal1/bert-tiny",
                    help="bert model type, it can be: prajjwal1/bert-tiny, yikuan8/Clinical-Longformer, etc.")
args = parser.parse_args()


def cli_main():

    all_auroc = []
    all_auprc = []
    all_f1 = []

    for seed in [41, 42, 43]:
        seed_everything(seed)

        # define datamodule
        if args.first_nrows == -1:
            args.first_nrows = None

        if args.task == "ihm":
            args.period_length = 48
        elif args.task == "pheno":
            args.period_length = 24

        if args.model_name == "ftlstm":
            args.batch_size = 1
        
        dm = MIMIC4DataModule(
            mimic_cxr_dir=str(MIMIC_CXR_JPG_PATH),
            file_path=str(
                DATA_PATH / f"output_mimic4/TS_CXR/{args.task}"),
            period_length=args.period_length,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            first_nrows=args.first_nrows,
            # num_imgs=args.num_imgs
            )
        
        # define model
        if args.test_only:
            args.devices = 1

        if args.model_name == "mtand":
            if args.ckpt_path:
                model = MIMIC4MTANDModule.load_from_checkpoint(
                    args.ckpt_path, **vars(args))
            else:
                model = MIMIC4MTANDModule(**vars(args))
        # elif args.model_name == "grud":
        #     if args.ckpt_path:
        #         model = GRUDModule.load_from_checkpoint(
        #             args.ckpt_path, **vars(args))
        #     else:
        #         model = GRUDModule(**vars(args))
        elif args.model_name == "flat":
            if args.ckpt_path:
                model = MIMIC4FlatModule.load_from_checkpoint(
                    args.ckpt_path, **vars(args))
            else:
                model = MIMIC4FlatModule(**vars(args))
        elif args.model_name == "hiertrans":
            if args.ckpt_path:
                model = MIMIC4HierTransformerModule.load_from_checkpoint(
                    args.ckpt_path, **vars(args))
            else:
                model = MIMIC4HierTransformerModule(**vars(args))
        elif args.model_name == "tlstm":
            if args.ckpt_path:
                model = MIMIC4TLSTMModule.load_from_checkpoint(
                    args.ckpt_path, **vars(args))
            else:
                model = MIMIC4TLSTMModule(**vars(args))
        # elif args.model_name == "ftlstm":
        #     # need to fix the batch size
        #     if args.ckpt_path:
        #         model = FTLSTMModule.load_from_checkpoint(
        #             args.ckpt_path, **vars(args))
        #     else:
        #         model = FTLSTMModule(**vars(args))
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
                    monitor="val_auroc",
                    mode="max",
                    save_top_k=2,
                    save_last=False),
                EarlyStopping(monitor="val_auroc", patience=5,
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
        all_f1.append(model.report_f1)

    report_df = pd.DataFrame({
        "auroc": all_auroc,
        "auprc": all_auprc,
        "f1": all_f1
    })

    mean_df = report_df.mean(axis=0)
    std_df = report_df.std(axis=0)
    statistic_df = pd.concat([mean_df, std_df], axis=1)
    statistic_df.columns = ["mean", "std"]
    print(statistic_df)


if __name__ == "__main__":
    cli_main()
