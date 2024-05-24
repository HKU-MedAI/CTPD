import pickle
from cv2 import transform
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
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from cmehr.paths import *


def get_transforms(is_train: bool = False):
    if is_train:
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((512, 512)),
            torchvision.transforms.RandomResizedCrop(512),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(0, 1)
            # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                                  std=[0.229, 0.224, 0.225])
        ])
    else:
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((512, 512)),
            torchvision.transforms.CenterCrop(512),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(0, 1)
            # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                                  std=[0.229, 0.224, 0.225])
        ])

    return transforms


def F_impute(X, tt, mask, duration, tt_max):
    ''' Impute missing values in time series by previous values.
    :param X: (n, 34)
    :param tt: (n,)
    :param mask: (n, 34)
    :param duration: int
    :param tt_max: int
    '''
    no_feature = X.shape[1]
    impute = np.zeros(shape=(tt_max//duration, no_feature*2))
    for x, t, m in zip(X, tt, mask):
        row = int(t/duration)
        if row >= tt_max:
            continue
        # iterate each feature
        for f_idx, (rwo_x, row_m) in enumerate(zip(x, m)):
            if row_m == 1:
                impute[row][no_feature+f_idx] = 1
                impute[row][f_idx] = rwo_x
            else:
                # For missing values:
                # TODO: if previous row has value, use it, else use 0 instead of normal values
                if impute[row-1][f_idx] != 0:
                    impute[row][f_idx] = impute[row-1][f_idx]

    return impute


class MIMIC4_Dataset(Dataset):
    def __init__(self,
                 mimic_cxr_dir: str,
                 file_path: str,
                 split: str,
                 img_transform=get_transforms(is_train=False),
                 modeltype: str = "TS_CXR",
                 tt_max: int = 48,
                 num_imgs: int = 5,
                 first_nrows: Optional[int] = None):
        super().__init__()

        data_path = os.path.join(file_path, f"{split}_p2x_data.pkl")
        with open(data_path, "rb") as f:
            self.data = pickle.load(f)

        self.mimic_cxr_dir = mimic_cxr_dir
        self.img_transform = img_transform
        assert self.img_transform != None, "Image transform is None"
        self.modeltype = modeltype
        self.tt_max = tt_max
        self.first_nrows = first_nrows
        self.num_imgs = num_imgs

        if self.first_nrows != None:
            self.data = self.data[:self.first_nrows]

        # filter data with more than 1000 irregular time steps
        print("Number of original samples: ", len(self.data))
        self.data = list(filter(lambda x: len(x['irg_ts']) < 1000, self.data))
        print(f"Number of filtered samples in {split} set: {len(self.data)}")

    def __getitem__(self, idx):
        data_detail = self.data[idx]
        idx = data_detail['name']
        reg_ts = data_detail['reg_ts']  # (48, 34)
        ts = data_detail['irg_ts']
        ts_mask = data_detail['irg_ts_mask']

        label = data_detail["label"]
        # text_time_to_end = data_detail["text_time_to_end"]
        ts_tt = data_detail["ts_tt"]
        # reg_ts = F_impute(ts, ts_tt, ts_mask, 1, self.tt_max)
        reg_ts = data_detail["reg_ts"]
        cxr_path = data_detail["cxr_path"]
        cxr_time = data_detail["cxr_time"]

        if "CXR" in self.modeltype:
            cxr_imgs = []
            for p in cxr_path:
                img = Image.open(os.path.join(
                    self.mimic_cxr_dir, p)).convert("RGB")
                img = self.img_transform(img)
                cxr_imgs.append(img)

        label = torch.tensor(label, dtype=torch.long)
        reg_ts = torch.tensor(reg_ts, dtype=torch.float)
        ts = torch.tensor(ts, dtype=torch.float)
        ts_mask = torch.tensor(ts_mask, dtype=torch.long)
        ts_tt = torch.tensor([t/self.tt_max for t in ts_tt], dtype=torch.float)
        cxr_time = [t/self.tt_max for t in cxr_time]
        cxr_time_mask = [1] * len(cxr_time)

        if 'CXR' in self.modeltype:
            if len(cxr_imgs) < self.num_imgs:
                num_pad_imgs = self.num_imgs - len(cxr_imgs)
                padded_imgs = [torch.zeros_like(cxr_imgs[0])] * num_pad_imgs
                cxr_imgs.extend(padded_imgs)
                cxr_time.extend([0] * num_pad_imgs)
                cxr_time_mask.extend([0] * num_pad_imgs)

            cxr_imgs = torch.stack(cxr_imgs)
            cxr_time = torch.tensor(cxr_time, dtype=torch.float)
            cxr_time_mask = torch.tensor(cxr_time_mask, dtype=torch.long)

        if 'CXR' not in self.modeltype:
            return {'idx': idx, 'ts': ts, 'ts_mask': ts_mask,
                    'ts_tt': ts_tt, 'reg_ts': reg_ts,
                    "label": label}

        else:
            return {'idx': idx, 'ts': ts, 'ts_mask': ts_mask,
                    'ts_tt': ts_tt, 'reg_ts': reg_ts,
                    "label": label,
                    "cxr_imgs": cxr_imgs[-self.num_imgs:],
                    "cxr_time": cxr_time[-self.num_imgs:],
                    "cxr_time_mask": cxr_time_mask[-self.num_imgs:]
                    }

    def __len__(self):
        return len(self.data)


def TSCXRIrgcollate_fn(batch):
    """ Collate fn for irregular time series and notes """

    ts_input_sequences = pad_sequence(
        [example['ts'] for example in batch], batch_first=True, padding_value=0)
    ts_mask_sequences = pad_sequence(
        [example['ts_mask'] for example in batch], batch_first=True, padding_value=0)
    ts_tt = pad_sequence([example['ts_tt']
                         for example in batch], batch_first=True, padding_value=0)
    label = torch.stack([example["label"] for example in batch])

    reg_ts_input = torch.stack([example['reg_ts'] for example in batch])
    if len(batch[0]) > 6:
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
            "cxr_time_mask": cxr_time_mask,
            "label": label
        }
    else:
        cxr_imgs, cxr_time, cxr_time_mask = None, None, None
        return {
            "ts": ts_input_sequences,
            "ts_mask": ts_mask_sequences,
            "ts_tt": ts_tt,
            "reg_ts": reg_ts_input,
            "label": label
        }


class MIMIC4DataModule(LightningDataModule):
    def __init__(self,
                 batch_size: int = 4,
                 num_workers: int = 1,
                 file_path: str = str(ROOT_PATH / "output/ihm"),
                 mimic_cxr_dir: str = str(MIMIC_CXR_JPG_PATH),
                 modeltype: str = "TS_CXR",
                 tt_max: int = 48,
                 first_nrows: Optional[int] = None
                 ) -> None:
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.file_path = file_path
        self.first_nrows = first_nrows
        self.mimic_cxr_dir = mimic_cxr_dir
        self.modeltype = modeltype
        self.tt_max = tt_max

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        dataset = MIMIC4_Dataset(
            mimic_cxr_dir=self.mimic_cxr_dir,
            file_path=self.file_path,
            split="train",
            img_transform=get_transforms(is_train=True),
            modeltype=self.modeltype,
            tt_max=self.tt_max,
            first_nrows=self.first_nrows
        )
        dataloader = DataLoader(dataset=dataset,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                shuffle=True,
                                collate_fn=TSCXRIrgcollate_fn)
        return dataloader

    def val_dataloader(self) -> TRAIN_DATALOADERS:
        dataset = MIMIC4_Dataset(
            mimic_cxr_dir=self.mimic_cxr_dir,
            file_path=self.file_path,
            split="val",
            img_transform=get_transforms(is_train=False),
            modeltype=self.modeltype,
            tt_max=self.tt_max,
            first_nrows=self.first_nrows
        )
        dataloader = DataLoader(dataset=dataset,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                shuffle=False,
                                collate_fn=TSCXRIrgcollate_fn)
        return dataloader

    def test_dataloader(self) -> TRAIN_DATALOADERS:
        dataset = MIMIC4_Dataset(
            mimic_cxr_dir=self.mimic_cxr_dir,
            file_path=self.file_path,
            split="test",
            img_transform=get_transforms(is_train=False),
            modeltype=self.modeltype,
            tt_max=self.tt_max,
            first_nrows=self.first_nrows
        )
        dataloader = DataLoader(dataset=dataset,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                shuffle=False,
                                collate_fn=TSCXRIrgcollate_fn)
        return dataloader


if __name__ == "__main__":
    dataset = MIMIC4_Dataset(
        mimic_cxr_dir=str(MIMIC_CXR_JPG_PATH),
        file_path=str(ROOT_PATH / "output_mimic4/TS_CXR/pheno"),
        split="val",
        img_transform=get_transforms(is_train=False),
        first_nrows=None
    )
    sample = dataset[48]

    # datamodule = MIMIC4DataModule(
    #     file_path=str(ROOT_PATH / "output_mimic4/TS_CXR/pheno"),
    #     tt_max=48
    # )
    # batch = dict()
    # for batch in datamodule.val_dataloader():
    #     break
    # for k, v in batch.items():
    #     print(f"{k}: ", v.shape)

    ipdb.set_trace()
    # """
    # ts: torch.Size([4, 72, 15])
    # ts_mask:  torch.Size([4, 72, 15])
    # ts_tt:  torch.Size([4, 72])
    # reg_ts:  torch.Size([4, 24, 30])
    # cxr_imgs: torch.size([4, 5, 3, 512, 512])
    # cxr_time: torch.Size([4, 5])
    # cxr_time_mask:  torch.Size([4, 5])
    # label: torch.Size([4])
    # """
