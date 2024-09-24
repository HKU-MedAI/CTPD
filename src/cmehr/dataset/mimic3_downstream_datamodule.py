import pickle
import ipdb
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
import numpy as np
from typing import Optional
import torch
from torch.nn.utils.rnn import pad_sequence
from lightning import LightningDataModule
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from cmehr.paths import *


class TSNote_Irg(Dataset):
    def __init__(self,
                 file_path: str,
                 split: str,
                 bert_type: str,
                 max_length: int,
                 modeltype: str = "TS_Text",
                 nodes_order: str = "Last",
                 num_of_notes: int = 5,
                 tt_max: int = 48,
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

        self.modeltype = modeltype
        self.bert_type = bert_type
        self.num_of_notes = num_of_notes
        self.tt_max = tt_max
        self.max_len = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(bert_type)
        self.first_nrows = first_nrows

        if self.first_nrows != None:
            self.data = self.data[:self.first_nrows]

        # filter data with more than 1000 irregular time steps
        print("Number of original samples: ", len(self.data))
        self.data = list(filter(lambda x: len(x['irg_ts']) < 1000, self.data))
        print(f"Number of filtered samples in {split} set: {len(self.data)}")

    def __getitem__(self, idx):
        if self.notes_order != None:
            notes_order = self.notes_order
        else:
            notes_order = 'Last' if self.order_sample[idx] == 1 else 'First'

        data_detail = self.data[idx]
        idx = data_detail['name']
        reg_ts = data_detail['reg_ts']  # (48, 34)
        ts = data_detail['irg_ts']
        ts_mask = data_detail['irg_ts_mask']

        # only keep samples with both time series and text data
        if 'text_data' not in data_detail:
            return None
        
        text = data_detail['text_data']

        if len(text) == 0:
            return None

        text_token = []
        atten_mask = []
        label = data_detail["label"]
        ts_tt = data_detail["ts_tt"]
        # text_time_to_end = data_detail["text_time_to_end"]
        text_time = data_detail["text_time"]
        # TODO: why do we need two reg_ts?

        if 'Text' in self.modeltype:
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

        label = torch.tensor(label, dtype=torch.long)
        reg_ts = torch.tensor(reg_ts, dtype=torch.float)
        ts = torch.tensor(ts, dtype=torch.float)
        ts_mask = torch.tensor(ts_mask, dtype=torch.long)
        ts_tt = torch.tensor([t/self.tt_max for t in ts_tt], dtype=torch.float)
        text_time = [t/self.tt_max for t in text_time]
        text_time_mask = [1] * len(text_time)

        if 'Text' in self.modeltype:
            while len(text_token) < self.num_of_notes:
                text_token.append(torch.tensor([0], dtype=torch.long))
                atten_mask.append(torch.tensor([0], dtype=torch.long))
                text_time.append(0)
                text_time_mask.append(0)
        text_time = torch.tensor(text_time, dtype=torch.float)
        text_time_mask = torch.tensor(text_time_mask, dtype=torch.long)

        if 'Text' not in self.modeltype:
            return {'idx': idx, 'ts': ts, 'ts_mask': ts_mask,
                    'ts_tt': ts_tt, 'reg_ts': reg_ts,
                    "label": label}

        if notes_order == "Last":
            return {'idx': idx, 'ts': ts, 'ts_mask': ts_mask,
                    'ts_tt': ts_tt, 'reg_ts': reg_ts,
                    "input_ids": text_token[-self.num_of_notes:],
                    "label": label,
                    "attention_mask": atten_mask[-self.num_of_notes:],
                    'note_time': text_time[-self.num_of_notes:],
                    'text_time_mask': text_time_mask[-self.num_of_notes:],
                    }
        else:
            return {'idx': idx, 'ts': ts, 'ts_mask': ts_mask,
                    'ts_tt': ts_tt, 'reg_ts': reg_ts,
                    "input_ids": text_token[:self.num_of_notes],
                    "label": label,
                    "attention_mask": atten_mask[:self.num_of_notes],
                    'note_time': text_time[:self.num_of_notes],
                    'text_time_mask': text_time_mask[:self.num_of_notes]
                    }

    def __len__(self):
        return len(self.data)


def TextTSIrgcollate_fn(batch):
    """ Collate fn for irregular time series and notes """
    batch = list(filter(lambda x: x is not None, batch))
    batch = list(filter(lambda x: len(x['ts']) < 1000, batch))  # type: ignore
    if len(batch) == 0:
        return None

    ts_input_sequences = pad_sequence(
        [example['ts'] for example in batch], batch_first=True, padding_value=0)
    ts_mask_sequences = pad_sequence(
        [example['ts_mask'] for example in batch], batch_first=True, padding_value=0)
    ts_tt = pad_sequence([example['ts_tt']
                         for example in batch], batch_first=True, padding_value=0)
    label = torch.stack([example["label"] for example in batch])

    reg_ts_input = torch.stack([example['reg_ts'] for example in batch])
    if len(batch[0]) > 6:
        input_ids = [pad_sequence(example['input_ids'], batch_first=True,
                                  padding_value=0).transpose(0, 1) for example in batch]
        attn_mask = [pad_sequence(example['attention_mask'], batch_first=True,
                                  padding_value=0).transpose(0, 1) for example in batch]

        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=0).transpose(1, 2)
        attn_mask = pad_sequence(
            attn_mask, batch_first=True, padding_value=0).transpose(1, 2)

        note_time = pad_sequence([example['note_time'].clone().detach().float()
                                  for example in batch], batch_first=True, padding_value=0)
        note_time_mask = pad_sequence([example['text_time_mask'].clone().detach().long()
                                      for example in batch], batch_first=True, padding_value=0)
    else:
        input_ids, attn_mask, note_time, note_time_mask = None, None, None, None

    return {
        "ts": ts_input_sequences,
        "ts_mask": ts_mask_sequences,
        "ts_tt": ts_tt,
        "reg_ts": reg_ts_input,
        "input_ids": input_ids,
        "attention_mask": attn_mask,
        "note_time": note_time,
        "note_time_mask": note_time_mask,
        "label": label
    }


class MIMIC3DataModule(LightningDataModule):
    def __init__(self,
                 batch_size: int = 4,
                 num_workers: int = 1,
                 modeltype: str = "TS_Text",
                 file_path: str = str(ROOT_PATH / "output/ihm"),
                #  bert_type: str = "yikuan8/Clinical-Longformer",
                 bert_type: str = "prajjwal1/bert-tiny",
                #  max_length: int = 1024,
                 max_length: int = 512, 
                 tt_max: int = 48,
                 first_nrows: Optional[int] = None
                 ) -> None:
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.file_path = file_path
        self.bert_type = bert_type
        self.max_length = max_length
        self.modeltype = modeltype
        self.first_nrows = first_nrows
        self.tt_max = tt_max

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        dataset = TSNote_Irg(
            file_path=self.file_path,
            split="train",
            bert_type=self.bert_type,
            modeltype=self.modeltype,
            max_length=self.max_length,
            tt_max=self.tt_max,
            first_nrows=self.first_nrows
        )
        dataloader = DataLoader(dataset=dataset,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                shuffle=True,
                                collate_fn=TextTSIrgcollate_fn)
        return dataloader

    def val_dataloader(self) -> TRAIN_DATALOADERS:
        dataset = TSNote_Irg(
            file_path=self.file_path,
            split="val",
            bert_type=self.bert_type,
            modeltype=self.modeltype,
            max_length=self.max_length,
            tt_max=self.tt_max,
            first_nrows=self.first_nrows
        )
        dataloader = DataLoader(dataset=dataset,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                shuffle=False,
                                collate_fn=TextTSIrgcollate_fn)
        return dataloader

    def test_dataloader(self) -> TRAIN_DATALOADERS:
        dataset = TSNote_Irg(
            file_path=self.file_path,
            split="test",
            bert_type=self.bert_type,
            max_length=self.max_length,
            modeltype=self.modeltype,
            tt_max=self.tt_max,
            first_nrows=self.first_nrows
        )
        dataloader = DataLoader(dataset=dataset,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                shuffle=False,
                                collate_fn=TextTSIrgcollate_fn)
        return dataloader


if __name__ == "__main__":
    dataset = TSNote_Irg(
        file_path=str(DATA_PATH / "output_mimic3/pheno"),
        split="train",
        bert_type="yikuan8/Clinical-Longformer",
        max_length=1024,
        modeltype="TS_Text",
        tt_max=48,
    )
    print(x)
    # datamodule = MIMIC3DataModule(
    #     file_path=str(DATA_PATH / "output_mimic3/pheno"),
    # )
    # batch = dict()
    # for batch in datamodule.val_dataloader():
    #     if batch is not None:
    #         break
    # for k, v in batch.items():
    #     print(f"{k}: ", v.shape)
    ipdb.set_trace()
    """
    ts: torch.Size([4, 157, 17])
    ts_mask:  torch.Size([4, 157, 17])
    ts_tt:  torch.Size([4, 157])
    reg_ts:  torch.Size([4, 48, 34])
    input_ids:  torch.Size([4, 5, 128])
    attention_mask:  torch.Size([4, 5, 128])
    note_time:  torch.Size([4, 5])
    note_time_mask: torch.Size([4, 5])
    label: torch.Size([4])
    """
