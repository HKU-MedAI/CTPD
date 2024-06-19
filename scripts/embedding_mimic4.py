import torch
from cmehr.models.common.model_PANTHER import PANTHER
import argparse
import os
import torch
from lightning import seed_everything
from torch.utils.data import DataLoader
from cmehr.utils.file_utils import save_pkl, load_pkl
from cmehr.utils.evaluation_utils import eval_svm, eval_linear
from cmehr.paths import *
import ipdb

'''
CUDA_VISIBLE_DEVICES=1 python embedding_mimic4.py
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
parser.add_argument("--proto_path", type=str, 
                    default=str(ROOT_PATH / "prototype_results/mimic4_pretrain/train_proto_50.pkl"))
args = parser.parse_args()


def cli_main():
    seed_everything(args.seed)
    
    if args.task in ["ihm", "readm"]:
        period_length = 48
    else:
        period_length = 24

    model = PANTHER(proto_path=args.proto_path).to(device)
    data_dict = load_pkl(os.path.join(args.emb_dir, f"{args.task}_embs.pkl"))
    train_ts_emb = data_dict["train_ts_embs"][:, :period_length]
    train_label = data_dict["train_label"]
    val_ts_emb = data_dict["val_ts_embs"][:, :period_length]
    val_label = data_dict["val_label"]
    test_ts_emb = data_dict["test_ts_embs"][:, :period_length]
    test_label = data_dict["test_label"]
    
    # For training set 
    train_loader = DataLoader(
        # train_ts_emb,
        list(zip(train_ts_emb, train_label)),
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        drop_last=False,
        shuffle=False)
    train_X, train_Y = model.predict(train_loader, use_cuda=torch.cuda.is_available())

    val_loader = DataLoader(
        # val_ts_emb,
        list(zip(val_ts_emb, val_label)),
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        drop_last=False,
        shuffle=False)
    val_X, val_Y = model.predict(val_loader, use_cuda=torch.cuda.is_available())

    # For test set
    test_loader = DataLoader(
        # test_ts_emb,
        list(zip(test_ts_emb, test_label)),
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
    save_pkl(os.path.join(args.emb_dir, f"{args.task}_ts_proto_embs.pkl"), embeddings)

    print(f"Evaluation method: {args.eval_method}")
    if args.eval_method == "svm":
        eval_svm(train_X, train_Y, test_X, test_Y)
    elif args.eval_method in ["linear", "mlp"]:
        eval_linear(train_X, train_Y, val_X, val_Y, test_X, test_Y,
                    batch_size=args.batch_size, task=args.task, 
                    eval_method=args.eval_method)


if __name__ == "__main__":
    cli_main()