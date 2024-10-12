import argparse
import os
import torch
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from lightning import seed_everything
from einops import rearrange
from cmehr.dataset.mimic4_pretraining_datamodule import MIMIC4MultimodalDataModule
from cmehr.dataset.mimic4_downstream_datamodule import MIMIC4DataModule
from cmehr.models.mimic4.stage1_pretrain_model import MIMIC4PretrainModule
from cmehr.paths import *
from cmehr.utils.file_utils import save_pkl, load_pkl
from cmehr.utils.evaluation_utils import eval_svm, eval_linear
import ipdb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True  # type: ignore
torch.backends.cudnn.benchmark = True      # type: ignore
torch.multiprocessing.set_sharing_strategy('file_system')
torch.set_float32_matmul_precision("high")


parser = argparse.ArgumentParser(description="Evaluate MIMIC IV")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--num_workers", type=int, default=2)
parser.add_argument("--first_nrows", type=int, default=-1)
parser.add_argument("--ckpt_path", type=str, 
                    default="")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--save_feat_dir", type=str, default=str(ROOT_PATH / "prototype_results"))
args = parser.parse_args()


@torch.no_grad()
def extract_pretrain_embs(model: MIMIC4PretrainModule, dataloader: DataLoader):
    # encode training data
    all_ts_embs = []
    all_cxr_embs = []
    all_name = []
    for _, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Encoding data"):
        # Get the embeddings for the time series data
        # ts = batch["ts"].to(device)
        # ts_mask = batch["ts_mask"].to(device)
        # ts_tt = batch["ts_tt"].to(device)
        # proj_ts_embs = model.forward_ts_mtand(ts, ts_mask, ts_tt)
        # proj_ts_embs = F.normalize(proj_ts_embs, dim=-1)
        all_name.extend(batch["name"])
        ts_dim = batch["reg_ts"].shape[2]
        reg_ts = batch["reg_ts"][..., :ts_dim//2].to(device)
        feat_ts = model.ts_conv1(reg_ts.permute(0, 2, 1))
        proj_ts_embs = model.ts_dilated_conv(feat_ts).permute(0, 2, 1)

        batch_size = reg_ts.size(0)
        reg_imgs = batch["reg_imgs"].to(device)
        reg_imgs = rearrange(reg_imgs, "b n c h w -> (b n) c h w")
        cxr_feats = model.img_encoder(reg_imgs).img_embedding
        cxr_embs = model.img_proj_layer(cxr_feats)
        cxr_embs = rearrange(cxr_embs, "(b n) d -> b n d", b=batch_size)
        # reg_imgs_mask = batch["reg_imgs_mask"].to(device)
        # cxr_mask = reg_imgs_mask.unsqueeze(-1).repeat(1, 1, cxr_embs.size(-1))
        # cxr_embs = torch.cat((cxr_embs, cxr_mask), 2)
        # img_feat = model.img_conv1(cxr_embs.permute(0, 2, 1))
        # proj_img_embs = model.img_dilated_conv(img_feat).permute(0, 2, 1)
        proj_img_embs = model.img_conv1(cxr_embs.permute(0, 2, 1)).permute(0, 2, 1)

        del reg_ts, feat_ts, reg_imgs, cxr_feats, cxr_embs

        # cxr_imgs = batch["cxr_imgs"].to(device)
        # cxr_time = batch["cxr_time"].to(device)
        # cxr_time_mask = batch["cxr_time_mask"].to(device)
        # proj_img_embs = model.extract_img_embs(cxr_imgs, cxr_time, cxr_time_mask)
        # # use the last time step
        # proj_img_embs = F.normalize(proj_img_embs, dim=-1)

        all_ts_embs.append(proj_ts_embs.cpu().numpy())
        all_cxr_embs.append(proj_img_embs.cpu().numpy())

    all_ts_embs = np.concatenate(all_ts_embs, axis=0)
    all_cxr_embs = np.concatenate(all_cxr_embs, axis=0)

    return all_ts_embs, all_cxr_embs, all_name


@torch.no_grad()
def extract_downstream_embs(model: MIMIC4PretrainModule, dataloader: DataLoader, task_period_length: int = 48):
    # encode training data
    all_ts_embs = []
    all_cxr_embs = []
    all_label = []
    all_name = []
    for _, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Encoding data"):

        # # Get the embeddings for the time series data
        # ts = batch["ts"].to(device)
        # ts_mask = batch["ts_mask"].to(device)
        # ts_tt = batch["ts_tt"].to(device)
        # proj_ts_embs = model.forward_ts_mtand(ts, ts_mask, ts_tt)
        # proj_ts_embs = F.normalize(proj_ts_embs, dim=-1)

        # cxr_imgs = batch["cxr_imgs"].to(device)
        # cxr_time = batch["cxr_time"].to(device)
        # cxr_time_mask = batch["cxr_time_mask"].to(device)
        # proj_img_embs = model.extract_img_embs(cxr_imgs, cxr_time, cxr_time_mask)
        # # use the last time step
        # proj_img_embs = F.normalize(proj_img_embs, dim=-1)

        all_name.extend(batch["name"])

        ts_dim = batch["reg_ts"].shape[2]
        reg_ts = batch["reg_ts"][..., :ts_dim//2].to(device)
        feat_ts = model.ts_conv1(reg_ts.permute(0, 2, 1))
        proj_ts_embs = model.ts_dilated_conv(feat_ts).permute(0, 2, 1)

        batch_size = reg_ts.size(0)
        reg_imgs = batch["reg_imgs"].to(device)
        reg_imgs = rearrange(reg_imgs, "b n c h w -> (b n) c h w")
        cxr_feats = model.img_encoder(reg_imgs).img_embedding
        cxr_embs = model.img_proj_layer(cxr_feats)
        cxr_embs = rearrange(cxr_embs, "(b n) d -> b n d", b=batch_size)
        # reg_imgs_mask = batch["reg_imgs_mask"].to(device)
        # cxr_mask = reg_imgs_mask.unsqueeze(-1).repeat(1, 1, cxr_embs.size(-1))
        # cxr_embs = torch.cat((cxr_embs, cxr_mask), 2)
        # img_feat = model.img_conv1(cxr_embs.permute(0, 2, 1))
        # proj_img_embs = model.img_dilated_conv(img_feat).permute(0, 2, 1)
        proj_img_embs = model.img_conv1(cxr_embs.permute(0, 2, 1)).permute(0, 2, 1)

        del reg_ts, feat_ts, reg_imgs, cxr_feats, cxr_embs

        all_ts_embs.append(proj_ts_embs.cpu().numpy())
        all_cxr_embs.append(proj_img_embs.cpu().numpy())
        all_label.append(batch["label"].cpu().numpy())

    all_ts_embs = np.concatenate(all_ts_embs, axis=0)
    all_cxr_embs = np.concatenate(all_cxr_embs, axis=0)
    all_label = np.concatenate(all_label, axis=0)

    return all_ts_embs, all_cxr_embs, all_label, all_name


def cli_main():
    seed_everything(args.seed)

    # define datamodule
    if args.first_nrows == -1:
        args.first_nrows = None
    args.save_feat_dir = os.path.join(BASE_DIR, args.save_feat_dir, f"mimic4_pretrain")
    os.makedirs(args.save_feat_dir, exist_ok=True)
    
    if args.ckpt_path:
        model = MIMIC4PretrainModule.load_from_checkpoint(args.ckpt_path, **vars(args))
    else:
        model = MIMIC4PretrainModule(**vars(args))
    model.eval()

    # Extract embeddings in the self-supervised multimodal dataset
    if not os.path.exists(os.path.join(args.save_feat_dir, "self_supervised_embs.pkl")):
        print("Extracting embeddings in the self-supervised multimodal dataset")
        dm = MIMIC4MultimodalDataModule(  
            mimic_cxr_dir=str(MIMIC_CXR_JPG_PATH),
            file_path=str(
                DATA_PATH / f"output_mimic4/self_supervised_multimodal"),
            period_length=48,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            first_nrows=args.first_nrows)
        train_ts_embs, train_cxr_embs, train_names = extract_pretrain_embs(model, dm.train_dataloader())
        save_dict = {
            "train_ts_embs": train_ts_embs,
            "train_cxr_embs": train_cxr_embs,
            "train_names": train_names
        }
        save_pkl(os.path.join(args.save_feat_dir, "self_supervised_embs.pkl"), save_dict)
    
    # Extract embeddings for IHM
    if not os.path.exists(os.path.join(args.save_feat_dir, "ihm_embs.pkl")):
        print("Extracting embeddings for IHM task")
        dm = MIMIC4DataModule(  
            mimic_cxr_dir=str(MIMIC_CXR_JPG_PATH),
            file_path=str(
                DATA_PATH / f"output_mimic4/TS_CXR/ihm"),
            # Here period length should be the same as the one used in the self-supervised learning task.
            period_length=48,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            first_nrows=args.first_nrows)
        train_ts_embs, train_cxr_embs, train_label, train_names = extract_downstream_embs(model, dm.train_dataloader(), task_period_length=48)
        val_ts_embs, val_cxr_embs, val_label, val_names = extract_downstream_embs(model, dm.val_dataloader(), task_period_length=48)
        test_ts_embs, test_cxr_embs, test_label, test_names = extract_downstream_embs(model, dm.test_dataloader(), task_period_length=48)
        save_dict = {
            "train_ts_embs": train_ts_embs,
            "train_cxr_embs": train_cxr_embs,
            "train_label": train_label,
            "train_names": train_names,
            "val_ts_embs": val_ts_embs,
            "val_cxr_embs": val_cxr_embs,
            "val_label": val_label,
            "val_names": val_names,
            "test_ts_embs": test_ts_embs,
            "test_cxr_embs": test_cxr_embs,
            "test_label": test_label,
            "test_names": test_names,
        }
        save_pkl(os.path.join(args.save_feat_dir, "ihm_embs.pkl"), save_dict)
    
    if not os.path.exists(os.path.join(args.save_feat_dir, "pheno_embs.pkl")):
        print("Extracting embeddings for PHENO task")
        dm = MIMIC4DataModule(  
            mimic_cxr_dir=str(MIMIC_CXR_JPG_PATH),
            file_path=str(
                DATA_PATH / f"output_mimic4/TS_CXR/pheno"),
            period_length=24,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            first_nrows=args.first_nrows)
        train_ts_embs, train_cxr_embs, train_label, train_names = extract_downstream_embs(model, dm.train_dataloader(), task_period_length=24)
        val_ts_embs, val_cxr_embs, val_label, val_names = extract_downstream_embs(model, dm.val_dataloader(), task_period_length=24)
        test_ts_embs, test_cxr_embs, test_label, test_names = extract_downstream_embs(model, dm.test_dataloader(), task_period_length=24)
        save_dict = {
            "train_ts_embs": train_ts_embs,
            "train_cxr_embs": train_cxr_embs,
            "train_label": train_label,
            "train_names": train_names,
            "val_ts_embs": val_ts_embs,
            "val_cxr_embs": val_cxr_embs,
            "val_label": val_label,
            "val_names": val_names,
            "test_ts_embs": test_ts_embs,
            "test_cxr_embs": test_cxr_embs,
            "test_label": test_label,
            "test_names": test_names,
        }
        save_pkl(os.path.join(args.save_feat_dir, "pheno_embs.pkl"), save_dict)
    
    # Evaluate the performance of the extracted embeddings
    for task in ["ihm", "pheno"]:   
        data_dict = load_pkl(os.path.join(args.save_feat_dir, f"{task}_embs.pkl"))
        # data_dict = load_pkl(os.path.join(args.save_feat_dir, "ihm_embs.pkl"))
        # use the first 48 time steps
        print("Evaluate TS embs: ")
        train_X = np.mean(data_dict["train_ts_embs"], axis=1)
        train_Y = data_dict["train_label"]
        val_X = np.mean(data_dict["val_ts_embs"], axis=1)
        val_Y = data_dict["val_label"]
        test_X = np.mean(data_dict["test_ts_embs"], axis=1)
        test_Y = data_dict["test_label"]
        eval_svm(train_X, train_Y, val_X, val_Y, test_X, test_Y)

        print("Evaluate CXR embs: ")
        train_X = np.mean(data_dict["train_cxr_embs"], axis=1)
        train_Y = data_dict["train_label"]
        val_X = np.mean(data_dict["val_cxr_embs"], axis=1)
        val_Y = data_dict["val_label"]
        test_X = np.mean(data_dict["test_cxr_embs"], axis=1)
        test_Y = data_dict["test_label"]
        eval_svm(train_X, train_Y, val_X, val_Y, test_X, test_Y)


if __name__ == "__main__":
    cli_main()