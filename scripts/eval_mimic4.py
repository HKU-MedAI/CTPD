import argparse
import torch
from tqdm import tqdm
from einops import rearrange
from lightning import Trainer, seed_everything
from cmehr.dataset import MIMIC4DataModule
from cmehr.models.mimic4.stage1_pretrain_model import MIMIC4PretrainModule
from sklearn.svm import SVC
import sklearn.metrics as metrics
from cmehr.paths import *
import ipdb

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
                    default="/home/fywang/Documents/EHR_codebase/MMMSPG/log/ckpts/mimic4_ihm_pretrain_2024-06-02_21-06-41/epoch=53-step=3672.ckpt")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()


@torch.no_grad()
def extract_embs(model, dataloader):
    # encode training data
    all_ts_embs = []
    all_cxr_embs = []
    all_label = []
    for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Encoding training data"):
        # Get the embeddings for the time series data
        ts = batch["ts"].to(device)
        ts_mask = batch["ts_mask"].to(device)
        ts_tt = batch["ts_tt"].to(device)
        ts_embs = model.forward_ts_mtand(ts, ts_mask, ts_tt)
        ts_embs = torch.mean(ts_embs, dim=1)

        cxr_imgs = batch["cxr_imgs"].to(device)
        cxr_time = batch["cxr_time"].to(device)
        cxr_time_mask = batch["cxr_time_mask"].to(device)
        batch_size = cxr_imgs.size(0)
        cxr_imgs = rearrange(cxr_imgs, "b n c h w -> (b n) c h w")
        cxr_feats = model.img_encoder(cxr_imgs).img_embedding
        cxr_embs = model.img_proj_layer(cxr_feats)
        cxr_embs = rearrange(cxr_embs, "(b n) d -> b n d", b=batch_size)
        cxr_sum_embs = (cxr_embs * cxr_time_mask.unsqueeze(-1)).sum(dim=1)
        cxr_mean_embs = cxr_sum_embs / cxr_time_mask.sum(dim=1, keepdim=True)

        all_ts_embs.append(ts_embs)
        all_cxr_embs.append(cxr_mean_embs)
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
    eval_ts_embs, eval_cxr_embs, eval_label = extract_embs(model, dm.test_dataloader())

    # SVM evaluation for TS 
    clf = SVC(C=1e6, gamma="scale")
    clf.fit(train_ts_embs, train_label)
    y_score = clf.decision_function(eval_ts_embs)
    auroc = metrics.roc_auc_score(eval_label, y_score)
    auprc = metrics.average_precision_score(eval_label, y_score)
    f1 = metrics.f1_score(eval_label, y_score > 0)
    print(f"[TS]\tAUROC: {auroc}, AUPRC: {auprc}, F1: {f1}")

    # SVM evaluation for CXR
    clf = SVC(C=1e6, gamma="scale")
    clf.fit(train_cxr_embs, train_label)
    y_score = clf.decision_function(eval_cxr_embs)
    auroc = metrics.roc_auc_score(eval_label, y_score)
    auprc = metrics.average_precision_score(eval_label, y_score)
    f1 = metrics.f1_score(eval_label, y_score > 0)
    print(f"[CXR]\tAUROC: {auroc}, AUPRC: {auprc}, F1: {f1}")
    

if __name__ == "__main__":
    cli_main()