'''
This script is used for self-supervised pretraining on multimodal MIMIC4 dataset.
'''
from argparse import ArgumentParser
from datetime import datetime
import ipdb

import torch
import os
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from cmehr.dataset.mimic4_pretraining_datamodule import MIMIC4MultimodalDataModule
from cmehr.models.mimic4.stage1_pretrain_model import MIMIC4PretrainModule
from cmehr.paths import *

torch.backends.cudnn.deterministic = True  # type: ignore
torch.backends.cudnn.benchmark = True  # type: ignore
torch.set_float32_matmul_precision("high")

'''
CUDA_VISIBLE_DEVICES=6,7 python pretrain_mimic4.py --devices 2
'''

parser = ArgumentParser(description="Self-supervised pretraining for MIMIC IV")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--max_epochs", type=int, default=100)
parser.add_argument("--devices", type=int, default=1)
parser.add_argument("--max_length", type=int, default=1024)
parser.add_argument("--accumulate_grad_batches", type=int, default=1)
parser.add_argument("--first_nrows", type=int, default=-1)
parser.add_argument("--ts_learning_rate", type=float, default=2e-5)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--period_length", type=int, default=48)
parser.add_argument("--num_imgs", type=int, default=4)
parser.add_argument("--cm_loss_weight", type=float, default=0.5)
args = parser.parse_args()


def cli_main():
    seed_everything(args.seed)

    # This is fixed for MIMIC4
    # args.orig_d_ts = 25
    # args.orig_reg_d_ts = 50
    args.orig_d_ts = 15
    args.orig_reg_d_ts = 30
    
    # define datamodule
    if args.first_nrows == -1:
        args.first_nrows = None

    # TODO: change this to use the task argument
    dm = MIMIC4MultimodalDataModule(  
        mimic_cxr_dir=str(MIMIC_CXR_JPG_PATH),
        file_path=str(
            DATA_PATH / f"output_mimic4/self_supervised_multimodal"),
        period_length=args.period_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        first_nrows=args.first_nrows,
        num_imgs=args.num_imgs
        )

    model = MIMIC4PretrainModule(**vars(args))
    model.train_iters_per_epoch = len(dm.train_dataloader()) // (args.accumulate_grad_batches * args.devices)
    # initialize trainer
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"mimic4_pretrain_{run_name}"
    os.makedirs(ROOT_PATH / "log/ckpts", exist_ok=True)
    logger = WandbLogger(
        name=run_name,
        save_dir=str(ROOT_PATH / "log"),
        project="cm-ehr", log_model=False)
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            dirpath=str(ROOT_PATH / "log/ckpts" / run_name),
            monitor="val_loss",
            mode="min",
            save_top_k=2,
            save_last=False),
        EarlyStopping(monitor="val_loss", patience=10,
                        mode="min", verbose=True)
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
        gradient_clip_val=0.5,
        # detect_anomaly=True
    )

    trainer.fit(model, dm)
    # trainer.test(model, datamodule=dm, ckpt_path="best")

    # close wandb
    import wandb
    wandb.finish()


if __name__ == "__main__":
    cli_main()
