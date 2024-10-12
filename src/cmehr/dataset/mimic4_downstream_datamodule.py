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
from cmehr.dataset.mimic4_pretraining_datamodule import get_transforms
from cmehr.paths import *


class MIMIC4_Dataset(Dataset):
    def __init__(self,
                 mimic_cxr_dir: str,
                 file_path: str,
                 split: str,
                 img_transform=get_transforms(is_train=False),
                 modeltype: str = "TS_CXR",
                 tt_max: int = 48,
                 num_imgs: int = 4,
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
        name = data_detail['name']
        # reg_ts = data_detail['reg_ts']  # (48, 34)
        # FIXME: This is a bug in the original code
        ts = data_detail['irg_ts']
        ts_mask = data_detail['irg_ts_mask']
        label = data_detail["label"]
        ts_tt = data_detail["ts_tt"]
        reg_ts = data_detail["reg_ts"]

        if "CXR" in self.modeltype:
            cxr_path = data_detail["cxr_path"]
            cxr_time = data_detail["cxr_time"]
            cxr_imgs = []
            for p in cxr_path:
                img = Image.open(os.path.join(
                    self.mimic_cxr_dir, p)).convert("RGB")
                img = self.img_transform(img)
                cxr_imgs.append(img)
            cxr_time = [t / self.tt_max for t in cxr_time]
            cxr_time_mask = [1] * len(cxr_time)

        label = torch.tensor(label, dtype=torch.long)
        reg_ts = torch.tensor(reg_ts, dtype=torch.float)
        ts = torch.tensor(ts, dtype=torch.float)
        ts_mask = torch.tensor(ts_mask, dtype=torch.long)
        # This is used to normalize the timestamps
        ts_tt = torch.tensor([t/self.tt_max for t in ts_tt], dtype=torch.float)

        if 'CXR' in self.modeltype:
            C, H, W = cxr_imgs[0].shape
            reg_img = torch.zeros(self.num_imgs, C, H, W)
            reg_img_mask = torch.zeros(self.num_imgs)
            cxr_time_indices = (np.array(cxr_time) * self.num_imgs).astype(int)
            previous_img = cxr_imgs[0]
            for i in range(self.num_imgs):
                cur_indices = np.where(cxr_time_indices == i)[0]
                if len(cur_indices) == 0:
                    reg_img[i] = previous_img
                else:
                    reg_img[i] = cxr_imgs[np.random.choice(cur_indices)]
                    previous_img = reg_img[i]
                    reg_img_mask[i] = 1

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
            return {'name': name, 'ts': ts, 'ts_mask': ts_mask,
                    'ts_tt': ts_tt, 'reg_ts': reg_ts,
                    "label": label}
        else:
            return {'name': name, 'ts': ts, 'ts_mask': ts_mask,
                    'ts_tt': ts_tt, 'reg_ts': reg_ts,
                    "label": label,
                    "cxr_imgs": cxr_imgs[-self.num_imgs:],
                    "cxr_time": cxr_time[-self.num_imgs:],
                    "cxr_time_mask": cxr_time_mask[-self.num_imgs:],
                    "reg_img": reg_img,
                    "reg_img_mask": reg_img_mask
                    }

    def __len__(self):
        return len(self.data)


def TSCXRIrgcollate_fn(batch):
    """ Collate fn for irregular time series and notes """
    name = [example['name'] for example in batch]
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
        reg_imgs = torch.stack([example['reg_img'] for example in batch])
        reg_imgs_mask = torch.stack([example['reg_img_mask'] for example in batch])
        return {
            "name": name,
            "ts": ts_input_sequences,
            "ts_mask": ts_mask_sequences,
            "ts_tt": ts_tt,
            "reg_ts": reg_ts_input,
            "cxr_imgs": cxr_imgs,
            "cxr_time": cxr_time,
            "cxr_time_mask": cxr_time_mask,
            "reg_imgs": reg_imgs,
            "reg_imgs_mask": reg_imgs_mask,
            "label": label
        }
    else:
        cxr_imgs, cxr_time, cxr_time_mask = None, None, None
        return {
            "name": name,
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
                 period_length: int = 48,
                 first_nrows: Optional[int] = None
                 ) -> None:
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.file_path = file_path
        self.first_nrows = first_nrows
        self.mimic_cxr_dir = mimic_cxr_dir
        self.modeltype = modeltype
        self.tt_max = period_length

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
    # dataset = MIMIC4_Dataset(
    #     mimic_cxr_dir=str(MIMIC_CXR_JPG_PATH),
    #     file_path=str(ROOT_PATH / "output_mimic4/TS_CXR/ihm"),
    #     split="val",
    #     first_nrows=None
    # )
    # sample = dataset[48]

    datamodule = MIMIC4DataModule(
        file_path=str(ROOT_PATH / "output_mimic4/TS_CXR/pheno"),
        period_length=48
    )
    batch = dict()
    for batch in datamodule.val_dataloader():
        break

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
