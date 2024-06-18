import argparse
import torch
from lightning import seed_everything
from cmehr.utils.file_utils import load_pkl
from cmehr.paths import *
from cmehr.utils.evaluation_utils import eval_svm, eval_linear
import ipdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True  # type: ignore
torch.backends.cudnn.benchmark = True  # type: ignore
torch.set_float32_matmul_precision("high")

'''
CUDA_VISIBLE_DEVICES=1 python eval_proto_mimic4.py
'''
parser = argparse.ArgumentParser(description="Evaluate MIMIC IV")
parser.add_argument("--eval_method", type=str, default="svm",
                    choices=["svm", "linear", "mlp"])
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--proto_emb_file", type=str, 
                    default="/home/fywang/Documents/EHR_codebase/MMMSPG/prototype_results/mimic4_pretrain/ihm_ts_proto_embs.pkl")
args = parser.parse_args()


def cli_main():
    seed_everything(42)
    args.task = "ihm"
    data_dict = load_pkl(args.proto_emb_file)
    train_X = data_dict["train_X"]
    train_y = data_dict["train_Y"]
    val_X = data_dict["val_X"]
    val_y = data_dict["val_Y"]
    test_X = data_dict["test_X"]
    test_y = data_dict["test_Y"]

    if args.eval_method == "svm":
        eval_svm(train_X, train_y, test_X, test_y)
    elif args.eval_method in ["linear", "mlp"]:
        eval_linear(train_X, train_y, val_X, val_y, test_X, test_y,
                    batch_size=args.batch_size, task=args.task, 
                    eval_method=args.eval_method)


if __name__ == "__main__":
    cli_main()