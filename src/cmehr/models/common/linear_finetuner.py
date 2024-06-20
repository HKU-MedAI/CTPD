import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn.metrics as metrics
from lightning import LightningModule
import ipdb
'''
These two functions are borrowed from: 
https://github.com/mahmoodlab/PANTHER/blob/main/src/mil_models/components.py.
'''
def create_mlp(in_dim=None, hid_dims=[], act=nn.ReLU(), dropout=0.,
               out_dim=None, end_with_fc=True, bias=True):

    layers = []
    if len(hid_dims) < 0:
        mlp = nn.Identity()
    elif len(hid_dims) >= 0:
        if len(hid_dims) > 0:
            for hid_dim in hid_dims:
                layers.append(nn.Linear(in_dim, hid_dim, bias=bias))
                layers.append(act)
                layers.append(nn.Dropout(dropout))
                in_dim = hid_dim
        layers.append(nn.Linear(in_dim, out_dim))
        if not end_with_fc:
            layers.append(act)
        mlp = nn.Sequential(*layers)
    return mlp


def create_mlp_with_dropout(in_dim=None, hid_dims=[], act=nn.ReLU(), dropout=0.,
               out_dim=None, end_with_fc=True, bias=True):

    layers = []
    if len(hid_dims) < 0:
        mlp = nn.Identity()
    elif len(hid_dims) >= 0:
        if len(hid_dims) > 0:
            for hid_dim in hid_dims:
                layers.append(nn.Linear(in_dim, hid_dim, bias=bias))
                layers.append(act)
                layers.append(nn.Dropout(dropout))
                in_dim = hid_dim
        layers.append(nn.Linear(in_dim, out_dim))
        if not end_with_fc:
            layers.append(act)
            layers.append(nn.Dropout(dropout))
        mlp = nn.Sequential(*layers)
    return mlp


class LinearFinetuner(LightningModule):
    def __init__(self, 
                 in_size: int,
                 num_classes: int = 2,
                 model_type: str = "linear",
                 n_proto: int = 100,
                 lr: float = 1e-3,
                 *args,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.in_size = in_size
        self.num_classes = num_classes
        self.model_type = model_type
        self.n_proto = n_proto
        if self.model_type == "linear":
            self.pred_layer = nn.Linear(self.in_size, self.num_classes)
        elif self.model_type == "mlp":
            # Individual embedding for prototypes
            shared_mlp = True
            indiv_mlps = True
            postcat_mlp = True
            in_dim = 257
            shared_embed_dim = 128
            indiv_embed_dim = 128
            postcat_embed_dim = 512
            n_fc_layers = 1
            shared_dropout = 0.1
            indiv_dropout = 0.1
            postcat_dropout = 0.1
            mlp_func = create_mlp_with_dropout
            if shared_mlp:
                self.shared_mlp = mlp_func(in_dim=in_dim,
                                           hid_dims=[shared_embed_dim] *
                                                    (n_fc_layers - 1),
                                            dropout=shared_dropout,
                                            out_dim=shared_embed_dim,
                                            end_with_fc=False)
                next_in_dim = shared_embed_dim
            else:
                self.shared_mlp = nn.Identity()
                next_in_dim = in_dim
            
            if indiv_mlps:
                self.indiv_mlps = nn.ModuleList([mlp_func(in_dim=next_in_dim,
                                                hid_dims=[indiv_embed_dim] *
                                                        (n_fc_layers - 1),
                                                dropout=indiv_dropout,
                                                out_dim=indiv_embed_dim,
                                                end_with_fc=False) for i in range(self.n_proto)])
                next_in_dim = self.n_proto * indiv_embed_dim
            else:
                self.indiv_mlps = nn.ModuleList([nn.Identity() for i in range (self.n_proto)])
                next_in_dim = self.n_proto * next_in_dim

            if postcat_mlp:
                self.postcat_mlp = mlp_func(in_dim=next_in_dim,
                                            hid_dims=[postcat_embed_dim] *
                                                    (n_fc_layers - 1),
                                            dropout=postcat_dropout,
                                            out_dim=postcat_embed_dim,
                                            end_with_fc=False)
                next_in_dim = postcat_embed_dim
            else:
                self.postcat_mlp = nn.Identity()
            
            self.classifier = nn.Linear(next_in_dim,
                                        num_classes,
                                        bias=False)
        
    def forward(self, x):
        if self.model_type == "linear":
            return self.pred_layer(x)
        elif self.model_type == "mlp":
            x = self.shared_mlp(x)
            x = torch.stack([self.indiv_mlps[idx](x[:, idx, :]) for idx in range(self.n_proto)], dim=1)
            x = x.reshape(x.shape[0], -1)   # (n_samples, n_proto * config.indiv_embed_dim)
            x = self.postcat_mlp(x)
            logits = self.classifier(x)
            return logits
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss)
        return loss

    def on_validation_epoch_start(self) -> None:
        self.val_step_outputs = []

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)    
        step_output = {
            "logits": logits,
            "y": y
        }
        self.val_step_outputs.append(step_output)

    def on_validation_epoch_end(self) -> None:
        if self.num_classes == 2:
            logits = torch.cat([x["logits"] for x in self.val_step_outputs], dim=0).detach().cpu().numpy()
            y = torch.cat([x["y"] for x in self.val_step_outputs], dim=0).detach().cpu().numpy()
            auroc = metrics.roc_auc_score(y, logits[:, 1])
            auprc = metrics.average_precision_score(y, logits[:, 1])
            f1 = metrics.f1_score(y, logits.argmax(axis=1))
            metrics_dict = {
                "val_auroc": auroc,
                "val_auprc": auprc,
                "val_f1": f1
            }
            self.log_dict(metrics_dict, on_epoch=True, on_step=False,
                        batch_size=len(y))
        else:
            raise NotImplementedError("Multi-class classification is not supported yet.")

    def on_test_epoch_start(self) -> None:
        self.test_step_outputs = []

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        step_output = {
            "logits": logits,
            "y": y
        }
        self.test_step_outputs.append(step_output)

    def on_test_epoch_end(self) -> None:
        if self.num_classes == 2:
            logits = torch.cat([x["logits"] for x in self.test_step_outputs], dim=0).detach().cpu().numpy()
            y = torch.cat([x["y"] for x in self.test_step_outputs], dim=0).detach().cpu().numpy()
            auroc = metrics.roc_auc_score(y, logits[:, 1])
            auprc = metrics.average_precision_score(y, logits[:, 1])
            f1 = metrics.f1_score(y, logits.argmax(axis=1))
            metrics_dict = {
                "test_auroc": auroc,
                "test_auprc": auprc,
                "test_f1": f1
            }
            self.log_dict(metrics_dict, on_epoch=True, on_step=False,
                        batch_size=len(y))
        else:
            raise NotImplementedError("Multi-class classification is not supported yet.")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=100, eta_min=1e-8
        )
        # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, factor=0.4, patience=3, verbose=True, mode='max')
        scheduler = {
            'scheduler': lr_scheduler,
            'monitor': 'val_auroc',
            'interval': 'epoch',
            'frequency': 1
        }
        return [optimizer], [scheduler]
    