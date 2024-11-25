import numpy as np
from PIL import Image
import math
import ot
import ipdb
from einops import rearrange
from typing import Optional, List
from torch import nn
from transformers import AutoModel, AutoConfig
import torch.nn.functional as F
import torch
from cmehr.models.mimic4.UTDE_modules import multiTimeAttention
from cmehr.models.mimic4.base_model import MIMIC4LightningModule
from cmehr.backbone.vision.pretrained import get_biovil_t_image_encoder
# from cmehr.models.mimic3.base_model import MIMIC3NoteModule
# from cmehr.models.mimic4.UTDE_modules import BertForRepresentation


class MIMIC4HierTransformerModule(MIMIC4LightningModule):
    def __init__(self,
                 task: str = "ihm",
                 modeltype: str = "CXR",
                 max_epochs: int = 10,
                 ts_learning_rate: float = 4e-4,
                 period_length: int = 48,
                 embed_dim: int = 128,
                #  bert_type: str = "prajjwal1/bert-tiny",
                 dropout: float = 0.1,
                 *args,
                 **kwargs
                 ):
        super().__init__(task=task, modeltype=modeltype, max_epochs=max_epochs,
                         ts_learning_rate=ts_learning_rate, period_length=period_length)
        self.save_hyperparameters()

        self.img_encoder = get_biovil_t_image_encoder()
        self.img_embed_dim = 512
        self.img_proj = nn.Linear(self.img_embed_dim, embed_dim)

        config = AutoConfig.from_pretrained("bert-base-uncased", hidden_size=128, num_hidden_layers=2,
                                             num_attention_heads=4)
        self.transformer = AutoModel.from_config(config)

        self.d_img = 128
        self.proj1 = nn.Linear(self.d_img, self.d_img)
        self.proj2 = nn.Linear(self.d_img, self.d_img)
        self.out_layer = nn.Linear(self.d_img, self.num_labels)
        self.dropout = dropout

    def forward(self,
                cxr_imgs: torch.Tensor,
                cxr_time: torch.Tensor,
                cxr_time_mask: torch.Tensor,
                labels=None,
                **kwargs):
        
        batch_size = cxr_imgs.size(0)
        flatten_imgs = rearrange(cxr_imgs, 'b n c h w -> (b n) c h w')
        x_img = self.img_encoder(flatten_imgs).img_embedding
        x_img = rearrange(x_img, '(b n) c -> b n c', b=batch_size)
        x_img = self.img_proj(x_img)

        last_hs = self.transformer(inputs_embeds=x_img).pooler_output
        # last_hs = torch.mean(x_txt, dim=1)
        # MLP for the final prediction
        last_hs_proj = self.proj2(
            F.dropout(F.relu(self.proj1(last_hs)), p=self.dropout, training=self.training))
        last_hs_proj += last_hs
        output = self.out_layer(last_hs_proj)

        if self.task in ['ihm', 'readm']:
            if labels != None:
                ce_loss = self.loss_fct1(output, labels)
                return ce_loss
            return F.softmax(output, dim=-1)[:, 1]

        elif self.task == 'pheno':
            if labels != None:
                labels = labels.float()
                ce_loss = self.loss_fct1(output, labels)
                return ce_loss
            return torch.sigmoid(output)


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from cmehr.paths import *
    # from cmehr.dataset.mimic4_pretraining_datamodule import MIMIC4DataModule
    # from cmehr.dataset.mimic3_downstream_datamodule import MIMIC3DataModule
    from cmehr.dataset.mimic4_downstream_datamodule import MIMIC4DataModule

    datamodule = MIMIC4DataModule(
        mimic_cxr_dir=str(MIMIC_CXR_JPG_PATH),
        file_path=str(DATA_PATH / "output_mimic4/TS_CXR/pheno"),
        period_length=24
    )
    batch = dict()
    for batch in datamodule.val_dataloader():
        break
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: ", v.shape)

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
    model = MIMIC4HierTransformerModule(task="pheno", period_length=24)
    loss = model(
        cxr_imgs=batch["cxr_imgs"],
        cxr_time=batch["cxr_time"],
        cxr_time_mask=batch["cxr_time_mask"],
        labels=batch["label"]
    )
    print(loss)

    # feat1 = torch.randn(12, 128)
    # feat2 = torch.randn(48, 128)
    # coattn = OT_Attn_assem(impl="pot-uot-l2", ot_reg=0.05, ot_tau=0.5)
    # A_coattn, dist = coattn(feat1, feat2)
    # A_coattn: optimal transport matrix feat1 -> feat2
