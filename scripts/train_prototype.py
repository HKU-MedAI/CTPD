'''
Train prototypes for Multimodal MIMIC-IV Dataset
CUDA_VISIBLE_DEVICES=0 python train_prototype.py
'''

from argparse import ArgumentParser
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import ipdb

from sklearn.cluster import KMeans
import torch
from cmehr.models.mimic4 import CNNModule
from cmehr.dataset import MIMIC4DataModule
from cmehr.paths import *

parser = ArgumentParser(description="Prototype learning.")
parser.add_argument("--task", type=str, default="ihm",
                    choices=["ihm", "decomp", "los", "pheno"])
parser.add_argument("--modeltype", type=str, default="TS",
                    choices=["TS_CXR", "TS", "CXR"])
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--max_length", type=int, default=1024)
parser.add_argument("--first_nrows", type=int, default=-1)
parser.add_argument("--save_dir", type=str, default=ROOT_PATH / "prototype_results")
parser.add_argument("--mode", type=str, default="faiss",
                    choices=["kmeans", "faiss"])
parser.add_argument("--ckpt_path", type=str, 
                    default="/home/fywang/Documents/MMMSPG/log/ckpts/mimic4_ihm_cnn_2024-05-28_16-07-43/epoch=25-step=884.ckpt")
parser.add_argument("--n_proto", type=int, default=20)
parser.add_argument("--n_iter", type=int, default=50)
parser.add_argument("--n_init", type=int, default=5)
args = parser.parse_args()


# def cluster(data_loader, n_proto, n_iter, n_init=5, feature_dim=1024,
#             mode="kmeans", use_cuda=False):

#     """
#     K-Means clustering on embedding space

#     For further details on FAISS,
#     https://github.com/facebookresearch/faiss/wiki/Faiss-building-blocks:-clustering,-PCA,-quantization
#     """
#     for batch in data_loader:
#         ipdb.set_trace()


def save_training_data(args):
    """
    Save the training data for clustering
    """

    dm = MIMIC4DataModule(
        mimic_cxr_dir=str(MIMIC_CXR_JPG_PATH),
        file_path=str(
            ROOT_PATH / f"output_mimic4/TS_CXR/{args.task}"),
        modeltype=args.modeltype,
        tt_max=args.period_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        first_nrows=args.first_nrows)
    data_loader = dm.train_dataloader()
    model = CNNModule.load_from_checkpoint(args.ckpt_path, **vars(args))
    model.eval()
    save_path = args.save_dir / f"mimic4_{args.task}_{args.modeltype}"

    all_ts_feat = []
    with torch.no_grad():
        for batch in tqdm(data_loader, total=len(data_loader), desc="Extracting TS features"):
            reg_ts = batch["reg_ts"].cuda()
            ts_feat = model.forward_feat(reg_ts)
            n_dim = ts_feat.size(-1)
            all_ts_feat.append(ts_feat.detach().cpu().reshape(-1, n_dim))
    all_ts_feat = torch.cat(all_ts_feat, dim=0)
    ipdb.set_trace()
    os.makedirs(save_path, exist_ok=True)
    torch.save(all_ts_feat, save_path / "ts_feat.pt")
    print(f"{len(all_ts_feat)} TS features saved at {save_path / 'ts_feat.pt'}")


def cluster(args):
    ts_feat_file = args.save_dir / f"mimic4_{args.task}_{args.modeltype}" / "ts_feat.pt"
    ts_feat = torch.load(ts_feat_file)
    print(f"Loaded TS features from {ts_feat_file}")
    if args.mode == "kmeans":
        pass
        # kmeans = KMeans(n_clusters=100, random_state=0).fit(ts_feat)
        # print(f"KMeans clustering done")
        # ipdb.set_trace()
    elif args.mode == "faiss":
        import faiss
        numOfGPUs = torch.cuda.device_count()
        print(f"\nUsing Faiss Kmeans for clustering with {numOfGPUs} GPUs...")
        print(f"\tNum of clusters {args.n_proto}, num of iter {args.n_iter}")
        kmeans = faiss.Kmeans(ts_feat.shape[1], 
                              args.n_proto, 
                              niter=args.n_iter, 
                              nredo=args.n_init,
                              verbose=True, 
                              max_points_per_centroid=len(ts_feat),
                              gpu=numOfGPUs)
        kmeans.train(ts_feat)
        weight = kmeans.centroids[np.newaxis, ...]
        ipdb.set_trace()


if __name__ == "__main__":
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

    save_training_data(args)
    cluster(args)