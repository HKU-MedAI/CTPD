'''
In this script, we study the multimodal fusion strategy of different methods.
CUDA_VISIBLE_DEVICES=0 python evaluate_multimodal.py
'''
import ipdb
import argparse
import torch
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, EarlyStopping
from cmehr.utils.file_utils import load_pkl
from cmehr.models.common.model_PANTHER import PrototypeTokenizer
from cmehr.models.common.multimodal_fusion import MultimodalFusion  
from cmehr.paths import *

parser = argparse.ArgumentParser(description="Evaluate multimodal fusion strategy.")
parser.add_argument("--task", type=str, default="ihm",
                    choices=["ihm", "decomp", "los", "pheno"])
parser.add_argument("--batch_size", type=int, default=48)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--proto_emb_dir", type=str, 
                    default=str(ROOT_PATH / "prototype_results/mimic4_pretrain"))
parser.add_argument("--n_proto", type=int, default=16)
args = parser.parse_args()
args.ts_proto_emb_path = f"{args.proto_emb_dir}/{args.task}_ts_proto_{args.n_proto}_embs.pkl"
args.cxr_proto_emb_path = f"{args.proto_emb_dir}/{args.task}_cxr_proto_{args.n_proto}_embs.pkl"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True  # type: ignore
torch.backends.cudnn.benchmark = True  # type: ignore
torch.set_float32_matmul_precision("high")


class CustomDataset(Dataset):
    def __init__(self, ts_embs, cxr_embs, label):
        self.ts_embs = ts_embs
        self.cxr_embs = cxr_embs
        self.label = label

    def __len__(self):
        return len(self.ts_embs) 

    def __getitem__(self, idx):
        return self.ts_embs[idx], self.cxr_embs[idx], self.label[idx]


def cli_main():
    ts_data_dict = load_pkl(args.ts_proto_emb_path)
    cxr_data_dict = load_pkl(args.cxr_proto_emb_path)

    n_proto = int(args.ts_proto_emb_path.split("/")[-1].replace(".pkl", "").replace("_embs", "").split("_")[-1])
    tokenizer = PrototypeTokenizer(p=n_proto)

    prob, mean, cov = tokenizer(ts_data_dict["train_X"])
    train_ts_X = torch.cat([torch.Tensor(prob).unsqueeze(dim=-1), torch.Tensor(mean), torch.Tensor(cov)], dim=-1)
    prob, mean, cov = tokenizer(ts_data_dict["val_X"])
    val_ts_X = torch.cat([torch.Tensor(prob).unsqueeze(dim=-1), torch.Tensor(mean), torch.Tensor(cov)], dim=-1)
    prob, mean, cov = tokenizer(ts_data_dict["test_X"])
    test_ts_X = torch.cat([torch.Tensor(prob).unsqueeze(dim=-1), torch.Tensor(mean), torch.Tensor(cov)], dim=-1)
    
    prob, mean, cov = tokenizer(cxr_data_dict["train_X"])
    train_cxr_X = torch.cat([torch.Tensor(prob).unsqueeze(dim=-1), torch.Tensor(mean), torch.Tensor(cov)], dim=-1)
    prob, mean, cov = tokenizer(cxr_data_dict["val_X"])
    val_cxr_X = torch.cat([torch.Tensor(prob).unsqueeze(dim=-1), torch.Tensor(mean), torch.Tensor(cov)], dim=-1)
    prob, mean, cov = tokenizer(cxr_data_dict["test_X"])
    test_cxr_X = torch.cat([torch.Tensor(prob).unsqueeze(dim=-1), torch.Tensor(mean), torch.Tensor(cov)], dim=-1)
    
    train_dataset = CustomDataset(train_ts_X, train_cxr_X, ts_data_dict["train_Y"])
    train_loader = DataLoader(train_dataset, 
                              batch_size=args.batch_size, 
                              num_workers=args.num_workers, 
                              shuffle=True)
    val_dataset = CustomDataset(val_ts_X, val_cxr_X, ts_data_dict["val_Y"])
    val_loader = DataLoader(val_dataset, 
                            batch_size=args.batch_size, 
                            num_workers=args.num_workers, 
                            shuffle=False)
    test_dataset = CustomDataset(test_ts_X, test_cxr_X, ts_data_dict["test_Y"])
    test_loader = DataLoader(test_dataset, 
                             batch_size=args.batch_size, 
                             num_workers=args.num_workers, 
                             shuffle=False)

    model = MultimodalFusion(
        in_ts_size=train_ts_X.shape[-1],
        in_cxr_size=train_cxr_X.shape[-1],
        shared_emb_dim=256,
        lr=2e-4)
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"mimic4_{args.task}_multimodal_{run_name}"
    logger = WandbLogger(
        name=run_name,
        save_dir=str(ROOT_PATH / "log"),
        project="cm-ehr", log_model=False)
    callbacks = [LearningRateMonitor(logging_interval="step"), 
                 EarlyStopping(monitor="val_auroc", mode="max", patience=10, verbose=True, min_delta=0.0)]
    trainer = Trainer(max_epochs=100, 
                      devices=1, 
                      deterministic=True,
                      callbacks=callbacks, 
                      logger=logger,
                      precision="16-mixed",
                      accelerator="gpu")
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader, ckpt_path="best")


if __name__ == "__main__":
    cli_main()