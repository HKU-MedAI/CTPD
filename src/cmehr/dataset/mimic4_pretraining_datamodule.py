import pickle
import ipdb
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
import numpy as np
from typing import Optional
# from cv2 import imread
from PIL import Image
import torch
import torchvision
from torch.nn.utils.rnn import pad_sequence
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from cmehr.paths import *


def get_transforms(is_train: bool = False):
    if is_train:
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((512, 512)),
            torchvision.transforms.RandomResizedCrop(512),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(0, 1)
        ])
    else:
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((512, 512)),
            torchvision.transforms.CenterCrop(512),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(0, 1)
        ])

    return transforms


class MIMIC4MultimodalDataset(Dataset):
    def __init__(self,
                 mimic_cxr_dir: str,
                 file_path: str,
                 split: str,
                 num_imgs: int = 12,
                 period_length: int = 400,
                 first_nrows: Optional[int] = None):
        super().__init__()
        data_path = os.path.join(file_path, f"norm_ts_{split}.pkl")
        with open(data_path, "rb") as f:
            self.data = pickle.load(f)

        self.mimic_cxr_dir = mimic_cxr_dir
        if split == "train":
            self.img_transform = get_transforms(is_train=True)
        else:
            self.img_transform = get_transforms(is_train=False)
        assert self.img_transform != None, "Image transform is None"
        self.first_nrows = first_nrows
        self.num_imgs = num_imgs
        self.ts_max = period_length

        print("Number of original samples: ", len(self.data))
        self.data = list(filter(lambda x: len(x['irg_ts']) < 1000, self.data))
        print(f"Number of filtered samples in {split} set: {len(self.data)}")
        
        if self.first_nrows != None:
            self.data = self.data[:self.first_nrows]

        # To remove the time steps greater than ts_max
        for sample in self.data:
            ts_indices = np.where(np.array(sample["ts_tt"]) <= self.ts_max)[0]
            if len(ts_indices) == 0:
                continue
            cxr_indices = np.where(np.array(sample["cxr_time"]) <= self.ts_max)[0]
            if len(cxr_indices) == 0:
                continue
            sample["ts_tt"] = np.array(sample["ts_tt"])[ts_indices]
            sample["irg_ts"] = np.array(sample["irg_ts"])[ts_indices]
            sample["irg_ts_mask"] = np.array(sample["irg_ts_mask"])[ts_indices]
            sample["cxr_time"] = np.array(sample["cxr_time"])[cxr_indices]
            sample["cxr_path"] = np.array(sample["cxr_path"])[cxr_indices]

    def __getitem__(self, idx):
        data_detail = self.data[idx]
        idx = data_detail['name']
        reg_ts = data_detail['reg_ts']  # (48, 34)
        ts = data_detail['irg_ts']
        ts_mask = data_detail['irg_ts_mask']

        ts_tt = data_detail["ts_tt"]
        reg_ts = data_detail["reg_ts"]
        cxr_path = data_detail["cxr_path"]
        cxr_time = data_detail["cxr_time"]

        cxr_imgs = []
        for p in cxr_path:
            img = Image.open(os.path.join(
                self.mimic_cxr_dir, p)).convert("RGB")
            img = self.img_transform(img)
            cxr_imgs.append(img)

        reg_ts = torch.tensor(reg_ts, dtype=torch.float)
        ts = torch.tensor(ts, dtype=torch.float)
        ts_mask = torch.tensor(ts_mask, dtype=torch.long)
        ts_tt = torch.tensor(ts_tt, dtype=torch.float) / self.ts_max
        cxr_time = [t / self.ts_max for t in cxr_time]
        cxr_time_mask = [1] * len(cxr_time)

        if len(cxr_imgs) < self.num_imgs:
            num_pad_imgs = self.num_imgs - len(cxr_imgs)
            padded_imgs = [torch.zeros_like(cxr_imgs[0])] * num_pad_imgs
            cxr_imgs.extend(padded_imgs)
            cxr_time.extend([0] * num_pad_imgs)
            cxr_time_mask.extend([0] * num_pad_imgs)

        cxr_imgs = torch.stack(cxr_imgs)
        cxr_time = torch.tensor(cxr_time, dtype=torch.float)
        cxr_time_mask = torch.tensor(cxr_time_mask, dtype=torch.long)

        if len(cxr_imgs) > self.num_imgs:
            # randomly choose num_imgs images
            indices = torch.randperm(len(cxr_imgs))[:self.num_imgs]
            sorted_indices, _ = indices.sort()
            cxr_imgs = cxr_imgs[sorted_indices]
            cxr_time = cxr_time[sorted_indices]
            cxr_time_mask = cxr_time_mask[sorted_indices]

        return {'idx': idx, 'ts': ts, 'ts_mask': ts_mask,
                'ts_tt': ts_tt, 'reg_ts': reg_ts,
                "cxr_imgs": cxr_imgs,
                "cxr_time": cxr_time,
                "cxr_time_mask": cxr_time_mask
                }

    def __len__(self):
        return len(self.data)


def custom_collate_fn(batch):
    """ Collate fn for irregular time series and notes """

    ts_input_sequences = pad_sequence(
        [example['ts'] for example in batch], batch_first=True, padding_value=0)
    ts_mask_sequences = pad_sequence(
        [example['ts_mask'] for example in batch], batch_first=True, padding_value=0)
    ts_tt = pad_sequence([example['ts_tt']
                         for example in batch], batch_first=True, padding_value=0)

    reg_ts_input = torch.stack([example['reg_ts'] for example in batch])
    cxr_imgs = torch.stack([example['cxr_imgs'] for example in batch])
    cxr_time = torch.stack(
        [example['cxr_time'] for example in batch])
    cxr_time_mask = torch.stack(
        [example['cxr_time_mask'] for example in batch])
    return {
        "ts": ts_input_sequences,
        "ts_mask": ts_mask_sequences,
        "ts_tt": ts_tt,
        "reg_ts": reg_ts_input,
        "cxr_imgs": cxr_imgs,
        "cxr_time": cxr_time,
        "cxr_time_mask": cxr_time_mask
    }


class MIMIC4MultimodalDataModule(LightningDataModule):
    def __init__(self,
                 batch_size: int = 4,
                 num_workers: int = 1,
                 file_path: str = str(ROOT_PATH / "output_mimic4/self_supervised_multimodal"),
                 mimic_cxr_dir: str = str(MIMIC_CXR_JPG_PATH),
                 modeltype: str = "TS_CXR",
                 period_length: int = 100,
                 first_nrows: Optional[int] = None
                 ) -> None:
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.file_path = file_path
        self.first_nrows = first_nrows
        self.mimic_cxr_dir = mimic_cxr_dir
        self.modeltype = modeltype
        self.period_length = period_length

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        dataset = MIMIC4MultimodalDataset(
            mimic_cxr_dir=self.mimic_cxr_dir,
            file_path=self.file_path,
            split="train",
            period_length=self.period_length,
            first_nrows=self.first_nrows
        )
        dataloader = DataLoader(dataset=dataset,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                shuffle=True,
                                collate_fn=custom_collate_fn)
        return dataloader

    def val_dataloader(self) -> TRAIN_DATALOADERS:
        dataset = MIMIC4MultimodalDataset(
            mimic_cxr_dir=self.mimic_cxr_dir,
            file_path=self.file_path,
            split="val",
            period_length=self.period_length,
            first_nrows=self.first_nrows
        )
        dataloader = DataLoader(dataset=dataset,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                shuffle=False,
                                collate_fn=custom_collate_fn)
        return dataloader

    def test_dataloader(self) -> TRAIN_DATALOADERS:
        dataset = MIMIC4MultimodalDataset(
            mimic_cxr_dir=self.mimic_cxr_dir,
            file_path=self.file_path,
            split="test",
            period_length=self.period_length,
            first_nrows=self.first_nrows
        )
        dataloader = DataLoader(dataset=dataset,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                shuffle=False,
                                collate_fn=custom_collate_fn)
        return dataloader
    


if __name__ == "__main__":
    # dataset = MIMIC4MultimodalDataset(
    #     mimic_cxr_dir=str(MIMIC_CXR_JPG_PATH),
    #     file_path=str(ROOT_PATH / "output_mimic4/self_supervised_multimodal"),
    #     split="val",
    #     first_nrows=None
    # )
    # sample = dataset[48]

    datamodule = MIMIC4MultimodalDataModule(
        file_path=str(ROOT_PATH / "output_mimic4/self_supervised_multimodal")
    )
    batch = dict()
    for batch in datamodule.val_dataloader():
        break
    for k, v in batch.items():
        print(f"{k}: ", v.shape)

    ipdb.set_trace()