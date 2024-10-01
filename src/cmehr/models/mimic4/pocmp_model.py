from typing import Dict
import ipdb
import math
import numpy as np
from einops import rearrange
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from lightning import LightningModule
from timm.models.layers import DropPath
from cmehr.models.mimic4.UTDE_modules import multiTimeAttention
# from cmehr.models.mimic4.tslanet_model import PatchEmbed, ICB
from cmehr.backbone.vision.pretrained import get_biovil_t_image_encoder
# from cmehr.utils.hard_ts_losses import hier_CL_hard
# from cmehr.utils.soft_ts_losses import hier_CL_soft
from cmehr.utils.lr_scheduler import linear_warmup_decay
from cmehr.models.common.dilated_conv import DilatedConvEncoder


class POCMPModule(LightningModule):
    '''
    The class of prototype-oriented contrastive multi-modal pretraining model.
    '''
    def __init__(self,
                 orig_d_ts: int = 15,
                 orig_reg_d_ts: int = 30,
                 warmup_epochs: int = 20,
                 max_epochs: int = 100,
                 ts_learning_rate: float = 4e-4,
                 embed_time: int = 64,
                 embed_dim: int = 128,
                 num_imgs: int = 5,
                 period_length: float = 100,
                 cm_loss_weight: float = 0.,
                 *args,
                 **kwargs
                 ):
        ''' Maybe to extract visual features offline ...'''
        # TODO: add more arguments for ablation study
        super().__init__()
        self.save_hyperparameters()
        
        self.orig_d_ts = orig_d_ts      
        self.orig_reg_d_ts = orig_reg_d_ts
        self.max_epochs = max_epochs
        self.ts_learning_rate = ts_learning_rate
        self.embed_dim = embed_dim
        self.num_imgs = num_imgs
        self.tt_max = period_length
        self.cm_loss_weight = cm_loss_weight
        self.warmup_epochs = warmup_epochs

        self.img_encoder = get_biovil_t_image_encoder()
        # Don't freeze image encoder
        # for param in self.img_encoder.parameters():
        #     param.requires_grad = False
        self.img_embed_dim = 512
        self.img_proj_layer = nn.Linear(self.img_embed_dim, self.embed_dim)

        '''
        change this into two mtand:
        - TS mtand: 48 time points
        - CXR mtand: 5 time points
        '''

        self.ts_conv1 = nn.Conv1d(self.orig_d_ts, self.embed_dim, kernel_size=1)
        self.img_conv1 = nn.Conv1d(self.embed_dim, self.embed_dim, kernel_size=1)

        # object-centric learning ... 
        # do we still need pretraing?
        depth = 1
        self.ts_dilated_conv = DilatedConvEncoder(
            in_channels=self.embed_dim, 
            channels=[self.embed_dim] * depth + [self.embed_dim], 
            kernel_size=3
        )



    def forward(self, x_ts, x_ts_mask, ts_tt_list,
                cxr_imgs,
                cxr_time,
                cxr_time_mask,
                labels=None, 
                reg_ts=None):
        pass

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from cmehr.paths import *
    # from cmehr.dataset.mimic4_pretraining_datamodule import MIMIC4DataModule
    from cmehr.dataset.mimic4_downstream_datamodule import MIMIC4DataModule

    datamodule = MIMIC4DataModule(
        file_path=str(DATA_PATH / "output_mimic4/TS_CXR/ihm"),
        period_length=48
    )
    batch = dict()
    for batch in datamodule.val_dataloader():
        break
    for k, v in batch.items():
        if isinstance(v, Tensor):
            print(f"{k}: ", v.shape)

    """
    ts: torch.Size([4, 157, 17])
    ts_mask:  torch.Size([4, 157, 17])
    ts_tt:  torch.Size([4, 157])
    reg_ts:  torch.Size([4, 48, 34])
    cxr_imgs:  torch.Size([4, 5, 3, 512, 512])
    cxr_time:  torch.Size([4, 5])
    cxr_time_mask:  torch.Size([4, 5])
    reg_imgs:  torch.Size([4, 5, 3, 512, 512])
    reg_imgs_mask:  torch.Size([4, 5])
    label: torch.Size([4])
    """
    model = POCMPModule(
    )
    loss = model(
        x_ts=batch["ts"],  # type ignore
        x_ts_mask=batch["ts_mask"],
        ts_tt_list=batch["ts_tt"],
        reg_ts=batch["reg_ts"],
        cxr_imgs=batch["input_ids"],
        labels=batch["label"],
    )
    print(loss)

    # feat1 = torch.randn(12, 128)
    # feat2 = torch.randn(48, 128)
    # coattn = OT_Attn_assem(impl="pot-uot-l2", ot_reg=0.05, ot_tau=0.5)
    # A_coattn, dist = coattn(feat1, feat2)
    # A_coattn: optimal transport matrix feat1 -> feat2
