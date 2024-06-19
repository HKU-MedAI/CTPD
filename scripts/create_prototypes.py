'''
Train prototypes for Multimodal MIMIC-IV Dataset
CUDA_VISIBLE_DEVICES=0 python train_prototype.py
'''

from argparse import ArgumentParser
import numpy as np
from einops import rearrange
import torch
from cmehr.utils.file_utils import save_pkl, load_pkl
from cmehr.paths import *

parser = ArgumentParser(description="Prototype learning.")
parser.add_argument("--task", type=str, default="ihm",
                    choices=["ihm", "decomp", "los", "pheno"])
parser.add_argument("--save_dir", type=str, default=ROOT_PATH / "prototype_results")
parser.add_argument("--mode", type=str, default="faiss",
                    choices=["kmeans", "faiss"])
parser.add_argument("--n_proto", type=int, default=50)
parser.add_argument("--n_iter", type=int, default=50)
parser.add_argument("--n_init", type=int, default=5)
args = parser.parse_args()


def cluster(args):
    pkl_file = args.save_dir / f"mimic4_pretrain/self_supervised_embs.pkl"
    data_dict = load_pkl(pkl_file)
    ts_feat = data_dict["train_ts_embs"]
    ts_feat = rearrange(ts_feat, "b n d -> (b n) d")

    print(f"Loaded TS features from {pkl_file}")
    if args.mode == "kmeans":
        pass
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
        save_proto_path = args.save_dir / f"mimic4_pretrain" / f"train_proto_{args.n_proto}.pkl"
        save_pkl(save_proto_path, {"prototypes": weight})


if __name__ == "__main__":
    cluster(args)