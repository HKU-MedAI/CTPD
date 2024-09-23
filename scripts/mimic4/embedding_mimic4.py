import torch
from cmehr.models.common.model_PANTHER import PANTHER, PrototypeTokenizer
import argparse
import os
import torch
import numpy as np
from lightning import seed_everything
from torch.utils.data import DataLoader
from cmehr.utils.file_utils import save_pkl, load_pkl
from cmehr.utils.evaluation_utils import eval_svm, eval_linear
from cmehr.paths import *
import ipdb

'''
CUDA_VISIBLE_DEVICES=3 python embedding_mimic4.py
'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True  # type: ignore
torch.backends.cudnn.benchmark = True  # type: ignore
torch.set_float32_matmul_precision("high")

parser = argparse.ArgumentParser(description="Evaluate MIMIC IV")
parser.add_argument("--task", type=str, default="ihm")
parser.add_argument("--eval_method", type=str, default="svm",
                    choices=["svm", "linear", "mlp"])
parser.add_argument("--batch_size", type=int, default=48)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--emb_dir", type=str,
                    default=str(ROOT_PATH / "prototype_results/mimic4_pretrain"))
parser.add_argument("--ts_proto_path", type=str, 
                    default=str(ROOT_PATH / "prototype_results/mimic4_pretrain/ts_proto_16.pkl"))
parser.add_argument("--cxr_proto_path", type=str, 
                    default=str(ROOT_PATH / "prototype_results/mimic4_pretrain/cxr_proto_16.pkl"))
args = parser.parse_args()


def save_agg_embs(train_emb, train_label, val_emb, val_label, test_emb, test_label, 
                  model, proto_emb_path):

    # For training set 
    train_loader = DataLoader(
        # train_emb,
        list(zip(train_emb, train_label)),
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        drop_last=False,
        shuffle=False)
    train_X, train_Y = model.predict(train_loader, use_cuda=torch.cuda.is_available())

    val_loader = DataLoader(
        # val_emb,
        list(zip(val_emb, val_label)),
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        drop_last=False,
        shuffle=False)
    val_X, val_Y = model.predict(val_loader, use_cuda=torch.cuda.is_available())

    # For test set
    test_loader = DataLoader(
        # test_emb,
        list(zip(test_emb, test_label)),
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        drop_last=False,
        shuffle=False)
    test_X, test_Y = model.predict(test_loader, use_cuda=torch.cuda.is_available())

    embeddings = {
        "train_X": train_X.cpu().numpy(),
        "train_Y": train_Y.cpu().numpy(),
        "val_X": val_X.cpu().numpy(),
        "val_Y": val_Y.cpu().numpy(),
        "test_X": test_X.cpu().numpy(),
        "test_Y": test_Y.cpu().numpy()
    }
    save_pkl(proto_emb_path, embeddings)


def evaluate_reps(args, train_X, train_Y, val_X, val_Y, test_X, test_Y, n_proto=16):
    print(f"Evaluation method: {args.eval_method}")
    if args.eval_method == "svm":
        eval_svm(train_X, train_Y, val_X, val_Y, test_X, test_Y)
    elif args.eval_method in ["linear", "mlp"]:
        if args.eval_method == "mlp":
            tokenizer = PrototypeTokenizer(p=n_proto)
            prob, mean, cov = tokenizer(train_X)
            train_X = torch.cat([torch.Tensor(prob).unsqueeze(dim=-1), torch.Tensor(mean), torch.Tensor(cov)], dim=-1)
            prob, mean, cov = tokenizer(val_X)
            val_X = torch.cat([torch.Tensor(prob).unsqueeze(dim=-1), torch.Tensor(mean), torch.Tensor(cov)], dim=-1)
            prob, mean, cov = tokenizer(test_X)
            test_X = torch.cat([torch.Tensor(prob).unsqueeze(dim=-1), torch.Tensor(mean), torch.Tensor(cov)], dim=-1)

        eval_linear(train_X, train_Y, val_X, val_Y, test_X, test_Y,
                    n_proto=n_proto, batch_size=args.batch_size, task=args.task, 
                    eval_method=args.eval_method)

def cli_main():
    seed_everything(args.seed)
    
    if args.task in ["ihm", "readm"]:
        period_length = 48
    else:
        period_length = 24

    data_dict = load_pkl(os.path.join(args.emb_dir, f"{args.task}_embs.pkl"))

    # For TS data
    n_proto = int(args.ts_proto_path.split("/")[-1].replace(".pkl", "").split("_")[-1])
    model = PANTHER(proto_path=args.ts_proto_path, out_size=n_proto).to(device)
    proto_emb_path = os.path.join(args.emb_dir, f"{args.task}_ts_proto_{n_proto}_embs.pkl")
    if not os.path.exists(proto_emb_path):
        save_agg_embs(data_dict["train_ts_embs"], data_dict["train_label"], 
                      data_dict["val_ts_embs"], data_dict["val_label"], 
                      data_dict["test_ts_embs"], data_dict["test_label"], 
                      model, proto_emb_path)

    loaded_data = load_pkl(proto_emb_path)
    train_X = loaded_data["train_X"]
    train_Y = loaded_data["train_Y"]
    val_X = loaded_data["val_X"]
    val_Y = loaded_data["val_Y"]
    test_X = loaded_data["test_X"]
    test_Y = loaded_data["test_Y"]
    evaluate_reps(args, train_X, train_Y, val_X, val_Y, test_X, test_Y, n_proto=n_proto)

    # For CXR data
    n_proto = int(args.cxr_proto_path.split("/")[-1].replace(".pkl", "").split("_")[-1])
    model = PANTHER(proto_path=args.cxr_proto_path, out_size=n_proto).to(device)
    proto_emb_path = os.path.join(args.emb_dir, f"{args.task}_cxr_proto_{n_proto}_embs.pkl")
    if not os.path.exists(proto_emb_path):
        save_agg_embs(data_dict["train_cxr_embs"], data_dict["train_label"], 
                      data_dict["val_cxr_embs"], data_dict["val_label"], 
                      data_dict["test_cxr_embs"], data_dict["test_label"], 
                      model, proto_emb_path)

    loaded_data = load_pkl(proto_emb_path)
    train_X = loaded_data["train_X"]
    train_Y = loaded_data["train_Y"]
    val_X = loaded_data["val_X"]
    val_Y = loaded_data["val_Y"]
    test_X = loaded_data["test_X"]
    test_Y = loaded_data["test_Y"]
    evaluate_reps(args, train_X, train_Y, val_X, val_Y, test_X, test_Y, n_proto=n_proto)

    # MM
    # n_proto = int(args.ts_proto_path.split("/")[-1].replace(".pkl", "").split("_")[-1])
    # model = PANTHER(proto_path=args.ts_proto_path, out_size=n_proto).to(device)
    # proto_emb_path = os.path.join(args.emb_dir, f"{args.task}_mm_proto_{n_proto}_embs.pkl")
    # if not os.path.exists(proto_emb_path):
    #     train_embs = np.concatenate([data_dict["train_ts_embs"], data_dict["train_cxr_embs"]], axis=1)
    #     val_embs = np.concatenate([data_dict["val_ts_embs"], data_dict["val_cxr_embs"]], axis=1)
    #     test_embs = np.concatenate([data_dict["test_ts_embs"], data_dict["test_cxr_embs"]], axis=1)
    #     save_agg_embs(train_embs, data_dict["train_label"], 
    #                   val_embs, data_dict["val_label"], 
    #                   test_embs, data_dict["test_label"], 
    #                   model, proto_emb_path)

    # loaded_data = load_pkl(proto_emb_path)
    # train_X = loaded_data["train_X"]
    # train_Y = loaded_data["train_Y"]
    # val_X = loaded_data["val_X"]
    # val_Y = loaded_data["val_Y"]
    # test_X = loaded_data["test_X"]
    # test_Y = loaded_data["test_Y"]
    # evaluate_reps(args, train_X, train_Y, val_X, val_Y, test_X, test_Y, n_proto=n_proto)


if __name__ == "__main__":
    cli_main()