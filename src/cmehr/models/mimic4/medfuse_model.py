import math
import ipdb
from torch import nn
import torch.nn.functional as F
import torch
from cmehr.models.mimic4.base_model import MIMIC4LightningModule
from cmehr.backbone.vision.pretrained import get_biovil_t_image_encoder
from cmehr.models.mimic4.UTDE_modules import multiTimeAttention


class MedFuseModule(MIMIC4LightningModule):
    def __init__(self,
                 task: str = "ihm",
                 modeltype: str = "TS_CXR",
                 max_epochs: int = 10,
                 img_learning_rate: float = 1e-4,
                 ts_learning_rate: float = 4e-4,
                 period_length: int = 48,
                 orig_d_ts: int = 15,
                 orig_reg_d_ts: int = 30,
                 reg_ts: bool = True,
                 irregular_learn_emb_cxr: bool = True,
                 pooling_type: str = "attention",
                 num_imgs: int = 5,
                 lamb: float = 1.,
                 embed_time: int = 64,
                 embed_dim: int = 128,
                 *args,
                 **kwargs
                 ):
        super().__init__(task=task, modeltype=modeltype, max_epochs=max_epochs,
                         img_learning_rate=img_learning_rate, ts_learning_rate=ts_learning_rate,
                         period_length=period_length)
        self.save_hyperparameters()

        self.reg_ts = reg_ts
        self.task = task
        self.tt_max = period_length
        self.orig_d_ts = orig_d_ts
        self.d_ts = embed_dim
        self.pooling_type = pooling_type
        self.lamb = lamb
        self.num_imgs = num_imgs

        if "CXR" in modeltype:
            self.irregular_learn_emb_cxr = irregular_learn_emb_cxr
            self.img_encoder = get_biovil_t_image_encoder()
            for param in self.img_encoder.parameters():
                param.requires_grad = False
            self.img_embed_dim = 512

            if self.irregular_learn_emb_cxr:
                # formulate the regular time stamps
                self.periodic = nn.Linear(1, embed_time-1)
                self.linear = nn.Linear(1, 1)
                self.time_query = torch.linspace(0, 1., self.tt_max)
                self.time_attn_cxr = multiTimeAttention(
                    self.img_embed_dim, embed_dim, embed_time, 8)

        # This is also irregular time series
        if "TS" in modeltype:
            self.ts_lstm = nn.LSTM(
                input_size=orig_reg_d_ts, hidden_size=self.d_ts, num_layers=1,
                batch_first=True, bidirectional=True)

        if modeltype == "TS_CXR":
            self.proj1 = nn.Linear(3 * self.d_ts, self.d_ts)
        elif modeltype == "TS":
            self.proj1 = nn.Linear(2 * self.d_ts, self.d_ts)
        elif modeltype == "CXR":
            self.proj1 = nn.Linear(self.d_ts, self.d_ts)
        else:
            raise NotImplementedError

        self.proj2 = nn.Linear(self.d_ts, self.d_ts)
        self.out_layer = nn.Linear(self.d_ts, self.num_labels)

    def learn_time_embedding(self, tt):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)

    def forward(self,
                x_ts=None,
                x_ts_mask=None,
                ts_tt_list=None,
                cxr_imgs_sequences=None,
                cxr_time_sequences=None,
                cxr_time_mask_sequences=None,
                labels=None,
                reg_ts=None):
        """ Forward function of Multimodal model
        :param x_ts: (B, N, D_t), torch.Tensor, time series data
        :param x_ts_mask: (B, N, D_t), torch.Tensor, time series mask
        :param ts_tt_list: (B, N), torch.Tensor, time series time
        :param labels: (B, ), torch.Tensor, labels
        :param reg_ts: (B, N_r, D_r), torch.Tensor, regular time series data
        """
        if "CXR" in self.modeltype:
            batch_size = cxr_imgs_sequences.size(0)
            valid_cxr_imgs = cxr_imgs_sequences[cxr_time_mask_sequences.bool()]
            cxr_feats = self.img_encoder(valid_cxr_imgs).img_embedding
            padded_feats = torch.zeros(
                batch_size, self.num_imgs, cxr_feats.size(-1)).type_as(cxr_feats)
            padded_feats[cxr_time_mask_sequences.bool()] = cxr_feats

            if self.irregular_learn_emb_cxr:
                time_key = self.learn_time_embedding(
                    cxr_time_sequences).to(self.device)
                time_query = self.learn_time_embedding(
                    self.time_query.unsqueeze(0)).to(self.device)
                proj_x_cxr = self.time_attn_cxr(
                    time_query, time_key, padded_feats, cxr_time_sequences)
                cxr_embs = proj_x_cxr.mean(dim=1)
            else:
                raise NotImplementedError

        if "TS" in self.modeltype:
            ts_feats, _ = self.ts_lstm(reg_ts)
            ts_embs = ts_feats[:, -1]  # batch_size, 256

        if self.modeltype == "TS_CXR":
            concat_embs = torch.cat([cxr_embs, ts_embs], dim=1)
            proj1 = F.relu(self.proj1(concat_embs))
        elif self.modeltype == "TS":
            proj1 = F.relu(self.proj1(ts_embs))
        elif self.modeltype == "CXR":
            proj1 = F.relu(self.proj1(cxr_embs))
        else:
            raise NotImplementedError

        proj2 = F.relu(self.proj2(proj1))
        output = self.out_layer(proj2)

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
    from cmehr.dataset.mimic4_pretraining_datamodule import MIMIC4DataModule

    datamodule = MIMIC4DataModule(
        file_path=str(ROOT_PATH / "output_mimic4/ihm"),
        tt_max=48
    )
    batch = dict()
    for batch in datamodule.val_dataloader():
        break
    for k, v in batch.items():
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
    model = MedFuseModule(
        modeltype="TS",
    )
    loss = model(
        x_ts=batch["ts"],  # type ignore
        x_ts_mask=batch["ts_mask"],
        ts_tt_list=batch["ts_tt"],
        cxr_imgs_sequences=batch["cxr_imgs"],
        cxr_time_sequences=batch["cxr_time"],
        cxr_time_mask_sequences=batch["cxr_time_mask"],
        reg_ts=batch["reg_ts"],
        labels=batch["label"],
    )
    print(loss)

    # feat1 = torch.randn(12, 128)
    # feat2 = torch.randn(48, 128)
    # coattn = OT_Attn_assem(impl="pot-uot-l2", ot_reg=0.05, ot_tau=0.5)
    # A_coattn, dist = coattn(feat1, feat2)
    # A_coattn: optimal transport matrix feat1 -> feat2
