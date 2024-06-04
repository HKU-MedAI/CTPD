import argparse
import os
import torch
from tqdm import tqdm
import torch.nn.functional as F
from lightning import seed_everything
from cmehr.dataset import MIMIC4DataModule
from cmehr.models.mimic4.stage1_pretrain_model import MIMIC4PretrainModule
from sklearn.svm import LinearSVC
import sklearn.metrics as metrics
from cmehr.paths import *
from cmehr.utils.file_utils import save_pkl
import ipdb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
'''
CUDA_VISIBLE_DEVICES=1 python eval_mimic4.py
'''
torch.backends.cudnn.deterministic = True  # type: ignore
torch.backends.cudnn.benchmark = True  # type: ignore
torch.set_float32_matmul_precision("high")


parser = argparse.ArgumentParser(description="Evaluate MIMIC IV")
parser.add_argument("--batch_size", type=int, default=48)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--first_nrows", type=int, default=-1)
parser.add_argument("--modeltype", type=str, default="TS_CXR",
                    choices=["TS_CXR", "TS", "CXR"],
                    help="Set the model type to use for training")
parser.add_argument("--ckpt_path", type=str, 
                    # default="/home/fywang/Documents/EHR_codebase/MMMSPG/log/ckpts/mimic4_ihm_pretrain_2024-06-03_15-05-02/epoch=98-step=6732.ckpt")
                    default="/home/fywang/Documents/EHR_codebase/MMMSPG/log/ckpts/mimic4_ihm_pretrain_2024-06-04_00-00-49/epoch=96-step=6596.ckpt")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--save_feat_dir", type=str, default="../prototype_results")
args = parser.parse_args()


@torch.no_grad()
def extract_embs(model, dataloader):
    # encode training data
    all_ts_embs = []
    all_cxr_embs = []
    all_label = []
    for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Encoding data"):
        # Get the embeddings for the time series data
        ts = batch["ts"].to(device)
        ts_mask = batch["ts_mask"].to(device)
        ts_tt = batch["ts_tt"].to(device)
        proj_ts_embs = model.forward_ts_mtand(ts, ts_mask, ts_tt)
        proj_ts_embs = F.normalize(proj_ts_embs, dim=-1)
        # ts_embs = proj_ts_embs[:, -1]

        cxr_imgs = batch["cxr_imgs"].to(device)
        cxr_time = batch["cxr_time"].to(device)
        cxr_time_mask = batch["cxr_time_mask"].to(device)
        proj_img_embs = model.extract_img_embs(cxr_imgs, cxr_time, cxr_time_mask)
        # use the last time step
        proj_img_embs = F.normalize(proj_img_embs, dim=-1)
        # img_embs = proj_img_embs[:, -1]

        all_ts_embs.append(proj_ts_embs)
        all_cxr_embs.append(proj_img_embs)
        # all_ts_embs.append(ts_embs)
        # all_cxr_embs.append(img_embs)
        all_label.append(batch["label"])

    all_ts_embs = torch.cat(all_ts_embs, dim=0).cpu().numpy()
    all_cxr_embs = torch.cat(all_cxr_embs, dim=0).cpu().numpy()
    all_label = torch.cat(all_label, dim=0).cpu().numpy()

    return all_ts_embs, all_cxr_embs, all_label


def cli_main():
    seed_everything(args.seed)

    # This is fixed for MIMIC4
    args.orig_d_ts = 15
    args.orig_reg_d_ts = 30

    # define datamodule
    if args.first_nrows == -1:
        args.first_nrows = None

    # TODO: change this to use the task argument
    args.task = "ihm"
    args.period_length = 48

    args.save_feat_dir = os.path.join(BASE_DIR, args.save_feat_dir, f"mimic4_ihm")
    os.makedirs(args.save_feat_dir, exist_ok=True)
    dm = MIMIC4DataModule(  
        mimic_cxr_dir=str(MIMIC_CXR_JPG_PATH),
        file_path=str(
            ROOT_PATH / f"output_mimic4/TS_CXR/{args.task}"),
        modeltype=args.modeltype,
        tt_max=args.period_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        first_nrows=args.first_nrows)
    
    if args.ckpt_path:
        model = MIMIC4PretrainModule.load_from_checkpoint(args.ckpt_path, **vars(args))
    else:
        model = MIMIC4PretrainModule(**vars(args))
    model.eval()

    train_ts_embs, train_cxr_embs, train_label = extract_embs(model, dm.train_dataloader())
    val_ts_embs, val_cxr_embs, val_label = extract_embs(model, dm.val_dataloader())
    test_ts_embs, test_cxr_embs, test_label = extract_embs(model, dm.test_dataloader())

    save_dict = {
        "train_ts_embs": train_ts_embs,
        "train_cxr_embs": train_cxr_embs,
        "train_label": train_label,
        "val_ts_embs": val_ts_embs,
        "val_cxr_embs": val_cxr_embs,
        "val_label": val_label,
        "test_ts_embs": test_ts_embs,
        "test_cxr_embs": test_cxr_embs,
        "test_label": test_label
    }
    save_pkl(os.path.join(args.save_feat_dir, "self_supervised_embs.pkl"), save_dict)

    # SVM evaluation for TS 
    pooling_method = "last"
    if pooling_method == "last":
        train_ts_embs_pool = train_ts_embs[:, -1]
        test_ts_embs_pool = test_ts_embs[:, -1]
        train_cxr_embs_pool = train_cxr_embs[:, -1]
        test_cxr_embs_pool = test_cxr_embs[:, -1]
    elif pooling_method == "mean":
        train_ts_embs_pool = train_ts_embs.mean(axis=1)
        test_ts_embs_pool = test_ts_embs.mean(axis=1)
        train_cxr_embs_pool = train_cxr_embs.mean(axis=1)
        test_cxr_embs_pool = test_cxr_embs.mean(axis=1)

    clf = LinearSVC(C=1e6, dual="auto")
    clf.fit(train_ts_embs_pool, train_label)
    y_score = clf.decision_function(test_ts_embs_pool)
    auroc = metrics.roc_auc_score(test_label, y_score)
    auprc = metrics.average_precision_score(test_label, y_score)
    f1 = metrics.f1_score(test_label, y_score > 0)
    print(f"[TS]\tAUROC: {auroc}, AUPRC: {auprc}, F1: {f1}")
    del clf
    
    clf = LinearSVC(C=1e6, dual="auto")
    clf.fit(train_cxr_embs_pool, train_label)
    y_score = clf.decision_function(test_cxr_embs_pool)
    auroc = metrics.roc_auc_score(test_label, y_score)
    auprc = metrics.average_precision_score(test_label, y_score)
    f1 = metrics.f1_score(test_label, y_score > 0)
    print(f"[CXR]\tAUROC: {auroc}, AUPRC: {auprc}, F1: {f1}")
    

if __name__ == "__main__":
    cli_main()