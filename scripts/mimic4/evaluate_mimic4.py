from argparse import ArgumentParser
import numpy as np
import ipdb
import sklearn.metrics as metrics

import torch
from cmehr.dataset.mimic4_downstream_datamodule import MIMIC4DataModule
from tqdm import tqdm
from cmehr.models.mimic4 import (CTPDModule)
from pprint import pprint
from cmehr.paths import *


torch.backends.cudnn.deterministic = True  # type: ignore
torch.backends.cudnn.benchmark = True  # type: ignore
torch.set_float32_matmul_precision("high")

'''
CUDA_VISIBLE_DEVICES=0 python evaluate_mimic4.py --task ihm
'''
parser = ArgumentParser()
parser.add_argument("--task", type=str, default="ihm")
parser.add_argument("--modeltype", type=str, default="TS_CXR")
parser.add_argument("--period_length", type=int, default=48)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--first_nrows", type=int, default=None)
parser.add_argument("--ckpt_path", type=str, 
                    default="/home/*/Documents/MMMSPG/log/ckpts/mimic4_ihm_CTPD_2024-10-02_23-12-01/epoch=46-step=4230.ckpt")
args = parser.parse_args()


@torch.no_grad()
def evaluate(model, val_loader, test_loader, task):
    val_logits = []
    val_labels = []
    for idx, batch in tqdm(enumerate(val_loader), total=len(val_loader), desc="Prediction on validation set"):
        batch = {k: v.to(model.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        logits = model(
                x_ts=batch["ts"],  # type ignore
                x_ts_mask=batch["ts_mask"],
                ts_tt_list=batch["ts_tt"],
                reg_ts=batch["reg_ts"],
                cxr_imgs=batch["cxr_imgs"],
                cxr_time=batch["cxr_time"],
                cxr_time_mask=batch["cxr_time_mask"],
                reg_imgs=batch["reg_imgs"],
                reg_imgs_mask=batch["reg_imgs_mask"],
            )
        labels = batch["label"]

        val_logits.append(logits.detach().cpu().numpy())
        val_labels.append(labels.detach().cpu().numpy())

    val_logits = np.concatenate(val_logits, axis=0)
    val_labels = np.concatenate(val_labels, axis=0)

    test_logits = []
    test_labels = []
    for idx, batch in tqdm(enumerate(test_loader), total=len(test_loader), desc="Prediction on test set"):
        batch = {k: v.to(model.device) for k, v in batch.items() if isinstance (v, torch.Tensor)}
        logits = model(
                x_ts=batch["ts"],  # type ignore
                x_ts_mask=batch["ts_mask"],
                ts_tt_list=batch["ts_tt"],
                reg_ts=batch["reg_ts"],
                cxr_imgs=batch["cxr_imgs"],
                cxr_time=batch["cxr_time"],
                cxr_time_mask=batch["cxr_time_mask"],
                reg_imgs=batch["reg_imgs"],
                reg_imgs_mask=batch["reg_imgs_mask"],
            )
        labels = batch["label"]

        test_logits.append(logits.detach().cpu().numpy())
        test_labels.append(labels.detach().cpu().numpy())

    test_logits = np.concatenate(test_logits, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)

    if task == "ihm":
        auroc = metrics.roc_auc_score(
            test_labels, test_logits)
        auprc = metrics.average_precision_score(
            test_labels, test_logits)
        _, _, thresholds = metrics.roc_curve(val_labels, val_logits)
        all_f1_scores = []
        for thres in thresholds:
            cur_pred = np.where(val_logits > thres, 1, 0)
            f1 = metrics.f1_score(val_labels, cur_pred)
            all_f1_scores.append(f1)
        best_thres = thresholds[np.argmax(all_f1_scores)]

        test_pred = np.where(test_logits > best_thres, 1, 0)
        test_f1 = metrics.f1_score(test_labels, test_pred)
    elif task == "pheno":
        auroc = metrics.roc_auc_score(test_labels, test_logits,
                                                average="macro")
        auprc = metrics.average_precision_score(
            test_labels, test_logits, average="macro")
        
        num_labels = test_labels.shape[1]
        cur_f1 = []
        for i in range(num_labels):
            _, _, thresholds = metrics.roc_curve(val_labels[:, i], val_logits[:, i])
            thresholds = thresholds[1:]
            all_f1_scores = []
            for thres in thresholds:
                cur_pred = np.where(val_logits[:, i] > thres, 1, 0)
                f1 = metrics.f1_score(val_labels[:, i], cur_pred)
                all_f1_scores.append(f1)
            best_thres = thresholds[np.argmax(all_f1_scores)]

            test_pred = np.where(test_logits[:, i] > best_thres, 1, 0)
            f1 = metrics.f1_score(test_labels[:, i], test_pred)
            cur_f1.append(f1)
        test_f1 = np.mean(cur_f1)

    metric_dict = {
        "AUROC": auroc,
        "AUPRC": auprc,
        "F1": test_f1,
    }
    pprint(metric_dict)


def main():
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

    model = CTPDModule.load_from_checkpoint(args.ckpt_path, map_location=torch.device("cuda:0"))
    model.eval()
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
    
    evaluate(model, dm.val_dataloader(), dm.test_dataloader(), args.task)


if __name__ == "__main__":
    main()