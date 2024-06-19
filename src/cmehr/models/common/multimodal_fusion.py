import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from cmehr.models.common.linear_finetuner import LinearFinetuner


class MultimodalFusion(LinearFinetuner):
    def __init__(self, in_ts_size, in_cxr_size, shared_emb_dim=128, 
                 num_classes=2, n_proto=100, lr=1e-3, *args, **kwargs):
        super().__init__(in_size=in_ts_size)
        self.save_hyperparameters()

        self.proj_ts = nn.Linear(in_ts_size, shared_emb_dim)
        self.proj_cxr = nn.Linear(in_cxr_size, shared_emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=shared_emb_dim, nhead=4, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # These layers can be replaced into MLP
        self.ts_post_layer = nn.Linear(shared_emb_dim, shared_emb_dim)
        self.cxr_post_layer = nn.Linear(shared_emb_dim, shared_emb_dim)
        self.classifier = nn.Linear(shared_emb_dim * 2, num_classes)

    def forward(self, ts_embs, cxr_embs):
        ts_embs = self.proj_ts(ts_embs)
        cxr_embs = self.proj_cxr(cxr_embs)
        embs = torch.cat([ts_embs, cxr_embs], dim=1)
        embs = self.transformer_encoder(embs)

        num_proto = embs.shape[1] // 2
        ts_feat = torch.mean(self.ts_post_layer(embs[:, :num_proto]), dim=1)
        cxr_feat = torch.mean(self.cxr_post_layer(embs[:, num_proto:]), dim=1)
        concat_feat = torch.cat([ts_feat, cxr_feat], dim=1)
        logits = self.classifier(concat_feat)

        return logits

    def training_step(self, batch, batch_idx):
        ts_emb, cxr_emb, y = batch
        logits = self(ts_emb, cxr_emb)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        ts_emb, cxr_emb, y = batch
        logits = self(ts_emb, cxr_emb)    
        step_output = {
            "logits": logits,
            "y": y
        }
        self.val_step_outputs.append(step_output)

    def test_step(self, batch, batch_idx):
        ts_emb, cxr_emb, y = batch
        logits = self(ts_emb, cxr_emb)
        step_output = {
            "logits": logits,
            "y": y
        }
        self.test_step_outputs.append(step_output)