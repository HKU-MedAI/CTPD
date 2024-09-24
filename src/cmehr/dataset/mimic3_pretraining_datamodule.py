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
from transformers import AutoTokenizer
from cmehr.paths import *


class MIMIC3MultimodalDataset(Dataset):
    def __init__(self,
                 file_path: str,
                 split: str,
                 bert_type: str,
                 max_length: int,
                 nodes_order: str = "Last",
                 num_of_notes: int = 5,
                 period_length: int = 48,
                 first_nrows: Optional[int] = None):
        super().__init__()

        data_path = os.path.join(file_path, f"{split}_p2x_data.pkl")
        with open(data_path, "rb") as f:
            self.data = pickle.load(f)

        if split == "train":
            self.notes_order = nodes_order
        else:
            self.notes_order = "Last"
        ratio_notes_order = None
        if ratio_notes_order != None:
            self.order_sample = np.random.binomial(
                1, ratio_notes_order, len(self.data))
            
        self.first_nrows = first_nrows
        self.ts_max = period_length
        self.max_len = max_length
        self.num_of_notes = num_of_notes

        print("Number of original samples: ", len(self.data))
        self.data = list(filter(lambda x: len(x['irg_ts']) < 1000, self.data))
        print(f"Number of filtered samples in {split} set: {len(self.data)}")
        
        if self.first_nrows != None:
            self.data = self.data[:self.first_nrows]

        self.bert_type = bert_type
        self.tokenizer = AutoTokenizer.from_pretrained(bert_type)

        # To remove the time steps greater than ts_max
        filtered_data = []
        for sample in self.data:
            ts_indices = np.where(np.array(sample["ts_tt"]) <= self.ts_max)[0]
            if len(ts_indices) == 0:
                continue
            text_indices = np.where(np.array(sample["text_time"]) <= self.ts_max)[0]
            if len(text_indices) == 0:
                continue
            sample["ts_tt"] = np.array(sample["ts_tt"])[ts_indices]
            sample["irg_ts"] = np.array(sample["irg_ts"])[ts_indices]
            sample["irg_ts_mask"] = np.array(sample["irg_ts_mask"])[ts_indices]
            sample["text_time"] = np.array(sample["text_time"])[text_indices]
            sample["text_data"] = np.array(sample["text_data"])[text_indices]
            filtered_data.append(sample)
        self.data = filtered_data

    def __getitem__(self, idx):
        data_detail = self.data[idx]
        idx = data_detail['name']
        reg_ts = data_detail['reg_ts']  # (48, 34)
        ts = data_detail['irg_ts']
        ts_mask = data_detail['irg_ts_mask']
        ts_tt = data_detail["ts_tt"]

        reg_ts = torch.tensor(reg_ts, dtype=torch.float)
        ts = torch.tensor(ts, dtype=torch.float)
        ts_mask = torch.tensor(ts_mask, dtype=torch.long)
        ts_tt = torch.tensor(ts_tt, dtype=torch.float) / self.ts_max
        text = data_detail['text_data']
        text_time = [t / self.ts_max for t in data_detail["text_time"]]
        text_time_mask = [1] * len(text)
 
        text_token = []
        atten_mask = []
        for t in text:
            inputs = self.tokenizer.encode_plus(t,
                                                padding="max_length",
                                                max_length=self.max_len,
                                                add_special_tokens=True,
                                                return_attention_mask=True,
                                                truncation=True)
            text_token.append(torch.tensor(
                inputs['input_ids'], dtype=torch.long))
            attention_mask = inputs['attention_mask']
            if "Longformer" in self.bert_type:
                attention_mask[0] += 1  # type: ignore
                atten_mask.append(torch.tensor(
                    attention_mask, dtype=torch.long))
            else:
                atten_mask.append(torch.tensor(
                    attention_mask, dtype=torch.long))
        
        L = text_token[0].shape[0]
        reg_text_token = torch.zeros(self.num_of_notes, L, dtype=torch.long)
        reg_atten_mask = torch.zeros(self.num_of_notes, L, dtype=torch.long)
        text_time_indices = (np.array(text_time) * self.num_of_notes).astype(int)
        previous_text_token = text_token[0]
        previous_atten_mask = atten_mask[0]
        for i in range(self.num_of_notes):
            cur_indices = np.where(text_time_indices == i)[0]
            if len(cur_indices) == 0:
                reg_text_token[i] = previous_text_token
                reg_atten_mask[i] = previous_atten_mask
            else:
                rand_idx = np.random.choice(cur_indices)
                reg_text_token[i] = text_token[rand_idx]
                reg_atten_mask[i] = atten_mask[rand_idx]
                previous_text_token = text_token[rand_idx]
                previous_atten_mask = atten_mask[rand_idx]

        if len(text_token) < self.num_of_notes:
            num_pad_text = self.num_of_notes - len(text_token)
            text_token.extend([torch.zeros(L, dtype=torch.long)] * num_pad_text)
            atten_mask.extend([torch.zeros(L, dtype=torch.long)] * num_pad_text)
            text_time.extend([0] * num_pad_text)
            text_time_mask.extend([0] * num_pad_text)
        
        text_token = torch.stack(text_token)
        atten_mask = torch.stack(atten_mask)
        text_time = torch.tensor(text_time, dtype=torch.float)
        text_time_mask = torch.tensor(text_time_mask, dtype=torch.long)

        if len(text_token) > self.num_of_notes:
            indices = torch.randperm(len(text_token))[:self.num_of_notes]
            sorted_indices, _ = indices.sort()
            text_token = text_token[sorted_indices]
            atten_mask = atten_mask[sorted_indices]
            text_time = text_time[sorted_indices]
            text_time_mask = text_time_mask[sorted_indices]

        return {'idx': idx, 
                'ts': ts, 
                'ts_mask': ts_mask, 
                'ts_tt': ts_tt,
                'reg_ts': reg_ts, 
                'input_ids': text_token, 
                'attention_mask': atten_mask,
                'text_time': text_time, 
                'text_time_mask': text_time_mask,
                'reg_input_ids': reg_text_token, 
                'reg_attention_mask': reg_atten_mask}

    def __len__(self):
        return len(self.data)


def custom_collate_fn(batch):
    """ Collate fn for irregular time series and notes """

    name = [example['idx'] for example in batch]
    ts_input_sequences = pad_sequence(
        [example['ts'] for example in batch], batch_first=True, padding_value=0)
    ts_mask_sequences = pad_sequence(
        [example['ts_mask'] for example in batch], batch_first=True, padding_value=0)
    ts_tt = pad_sequence([example['ts_tt']
                         for example in batch], batch_first=True, padding_value=0)

    reg_ts_input = torch.stack([example['reg_ts'] for example in batch])

    text_token = pad_sequence(
        [example['input_ids'] for example in batch], batch_first=True, padding_value=0)
    atten_mask = pad_sequence(
        [example['attention_mask'] for example in batch], batch_first=True, padding_value=0)
    text_time = torch.stack([example['text_time'] for example in batch])
    text_time_mask = torch.stack(
        [example['text_time_mask'] for example in batch])
    reg_text_token = pad_sequence(
        [example['reg_input_ids'] for example in batch], batch_first=True, padding_value=0)
    reg_atten_mask = pad_sequence(
        [example['reg_attention_mask'] for example in batch], batch_first=True, padding_value=0)

    return {
        'name': name,
        'ts': ts_input_sequences,
        'ts_mask': ts_mask_sequences,
        'ts_tt': ts_tt,
        'reg_ts': reg_ts_input,
        'input_ids': text_token, 
        'attention_mask': atten_mask,
        'text_time': text_time, 
        'text_time_mask': text_time_mask,
        'reg_input_ids': reg_text_token, 
        'reg_attention_mask': reg_atten_mask
    }

class MIMIC3MultimodalDataModule(LightningDataModule):
    def __init__(self,
                 batch_size: int = 4,
                 num_workers: int = 1,
                 file_path: str = str(ROOT_PATH / "output_mimic4/self_supervised_multimodal"),
                 modeltype: str = "TS_Text",
                 bert_type: str = "prajjwal1/bert-tiny",
                 max_length: int = 512,
                 period_length: int = 100,
                 first_nrows: Optional[int] = None
                 ) -> None:
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.file_path = file_path
        self.first_nrows = first_nrows
        self.modeltype = modeltype
        self.period_length = period_length
        self.bert_type = bert_type
        self.max_length = max_length

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        dataset = MIMIC3MultimodalDataset(
            file_path=self.file_path,
            split="train",
            bert_type=self.bert_type,
            max_length=self.max_length,
            first_nrows=None
        )
        dataloader = DataLoader(dataset=dataset,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                shuffle=True,
                                drop_last=True,
                                collate_fn=custom_collate_fn)
        return dataloader

    def val_dataloader(self) -> TRAIN_DATALOADERS:
        dataset = MIMIC3MultimodalDataset(
            file_path=self.file_path,
            split="val",
            bert_type=self.bert_type,
            max_length=self.max_length,
            first_nrows=None
        )
        dataloader = DataLoader(dataset=dataset,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                shuffle=False,
                                drop_last=True,
                                collate_fn=custom_collate_fn)
        return dataloader

    def test_dataloader(self) -> TRAIN_DATALOADERS:
        dataset = MIMIC3MultimodalDataset(
            file_path=self.file_path,
            split="test",
            bert_type=self.bert_type,
            max_length=self.max_length,
            first_nrows=None
        )
        dataloader = DataLoader(dataset=dataset,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                shuffle=False,
                                drop_last=True,
                                collate_fn=custom_collate_fn)
        return dataloader
    


if __name__ == "__main__":
    # dataset = MIMIC3MultimodalDataset(
    #     file_path=str(ROOT_PATH / "output_mimic3/self_supervised_multimodal"),
    #     split="val",
    #     bert_type="yikuan8/Clinical-Longformer",
    #     max_length=1024,
    #     first_nrows=None
    # )
    # sample = dataset[1]
    # ipdb.set_trace()

    datamodule = MIMIC3MultimodalDataModule(
        file_path=str(ROOT_PATH / "output_mimic3/self_supervised_multimodal"),
        batch_size=4,
    )
    batch = dict()
    for batch in datamodule.train_dataloader():
        break

    for k, v in batch.items():
        if k == "name":
            print(f"{k}: ", v)
        else:
            print(f"{k}: ", v.shape)
    # check if there are nan in reg_ts
    # pkl_file = ROOT_PATH / "output_mimic3/self_supervised_multimodal" / "norm_ts_train.pkl"
    # from tqdm import tqdm
    # with open(pkl_file, "rb") as f:
    #     data = pickle.load(f)
    #     for sample in tqdm(data, total=len(data)):
    #         if np.isnan(sample["reg_ts"]).any():
    #             print("Nan in reg_ts")
    #             break

        # for k, v in batch.items():
        #     print(f"{k}: ", v.shape)
        #     print(batch["reg_ts"].isnan().any()) 