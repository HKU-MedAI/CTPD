'''
Train prototypes for Multimodal MIMIC-IV Dataset
CUDA_VISIBLE_DEVICES=0 python train_prototype.py
'''
import ipdb
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
parser.add_argument("--n_proto", type=int, default=16)
parser.add_argument("--n_iter", type=int, default=50)
parser.add_argument("--n_init", type=int, default=5)
args = parser.parse_args()


def cluster(args, feat, save_proto_path):
    if args.mode == "kmeans":
        pass
    elif args.mode == "faiss":
        import faiss
        numOfGPUs = torch.cuda.device_count()
        print(f"\nUsing Faiss Kmeans for clustering with {numOfGPUs} GPUs...")
        print(f"\tNum of clusters {args.n_proto}, num of iter {args.n_iter}")
        kmeans = faiss.Kmeans(feat.shape[1], 
                              args.n_proto,
                              niter=args.n_iter,
                              nredo=args.n_init,
                              verbose=True, 
                              max_points_per_centroid=len(feat),
                              gpu=numOfGPUs)
        kmeans.train(feat)
        weight = kmeans.centroids[np.newaxis, ...]
        save_pkl(save_proto_path, {"prototypes": weight})


if __name__ == "__main__":
    pkl_file = args.save_dir / f"mimic4_pretrain/self_supervised_embs.pkl"
    data_dict = load_pkl(pkl_file)
    print(f"Loaded TS features from {pkl_file}")
    ts_feat = data_dict["train_ts_embs"]
    ts_feat = rearrange(ts_feat, "b n d -> (b n) d")
    cxr_feat = data_dict["train_cxr_embs"]
    cxr_feat = rearrange(cxr_feat, "b n d -> (b n) d")

    print(f"Loaded TS features from {pkl_file}")
    save_proto_path = args.save_dir / f"mimic4_pretrain" / f"ts_proto_{args.n_proto}.pkl"
    cluster(args, ts_feat, save_proto_path)

    print(f"Loaded CXR features from {pkl_file}")
    save_proto_path = args.save_dir / f"mimic4_pretrain" / f"cxr_proto_{args.n_proto}.pkl"
    cluster(args, cxr_feat, save_proto_path)

    # # TODO: another idea is to use all embeddings for clustering
    # feat = np.concatenate([ts_feat, cxr_feat], axis=0)
    # save_proto_path = args.save_dir / f"mimic4_pretrain" / f"mm_proto_{args.n_proto}.pkl"
    # cluster(args, feat, save_proto_path)