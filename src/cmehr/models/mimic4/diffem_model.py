import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

from cmehr.models.mimic4.base_model import MIMIC4LightningModule
from cmehr.backbone.time_series.inceptiontime import InceptionTimeFeatureExtractor
from cmehr.backbone.time_series.resnet import ResNetFeatureExtractor
from cmehr.backbone.time_series.pooling import MILConjunctivePooling

import ipdb



def load_pkl(filename):
	loader = open(filename,'rb')
	file = pickle.load(loader)
	loader.close()
	return file


def mog_eval(mog, data):
    """
    This evaluates the log-likelihood of mixture of Gaussians
    """    
    B, N, d = data.shape    
    pi, mu, Sigma = mog    
    if len(pi.shape)==1:
        pi = pi.unsqueeze(0).repeat(B,1)
        mu = mu.unsqueeze(0).repeat(B,1,1)
        Sigma = Sigma.unsqueeze(0).repeat(B,1,1)
    p = pi.shape[-1]
    
    jll = -0.5 * ( d * np.log(2*np.pi) + 
        Sigma.log().sum(-1).unsqueeze(1) +
        torch.bmm(data**2, 1./Sigma.permute(0,2,1)) + 
        ((mu**2) / Sigma).sum(-1).unsqueeze(1) + 
        -2. * torch.bmm(data, (mu/Sigma).permute(0,2,1))
    ) + pi.log().unsqueeze(1) 
    
    mll = jll.logsumexp(-1) 
    cll = jll - mll.unsqueeze(-1)
    
    return jll, cll, mll


class DirNIWNet(nn.Module):
    """
    Conjugate prior for the Gaussian mixture model

    Args:
    - p (int): Number of prototypes
    - d (int): Embedding dimension
    - eps (float): initial covariance (similar function to sinkorn entropic regularizer)
    """
    
    def __init__(self, p, d, eps=0.1, load_proto=True, proto_path=None, fix_proto=True):
        """
        self.m: prior mean (p x d)
        self.V_: prior covariance (diagonal) (p x d)
        """
        super(DirNIWNet, self).__init__()

        self.load_proto = load_proto
        self.eps = eps

        if self.load_proto:
            if proto_path.endswith('pkl'):
                weights = load_pkl(proto_path)['prototypes'].squeeze()
            elif proto_path.endswith('npy'):
                weights = np.load(proto_path)

            self.m = nn.Parameter(torch.from_numpy(weights), requires_grad=not fix_proto)
        else:
            self.m = nn.Parameter(0.1 * torch.randn(p, d), requires_grad=not fix_proto)

        self.V_ = nn.Parameter(np.log(np.exp(1) - 1) * torch.ones((p, d)), requires_grad=not fix_proto)

        self.p, self.d = p, d
    
    def forward(self):
        """
        Return prior mean and covariance
        """
        V = self.eps * F.softplus(self.V_)
        return self.m, V
    
    def mode(self, prior=None):
        if prior is None:
            m, V = self.forward()
        else:
            m, V = prior
        pi = torch.ones(self.p).to(m) / self.p
        mu = m
        Sigma = V
        return pi.float(), mu.float(), Sigma.float()
    
    def loglik(self, theta): 
        raise NotImplementedError
        
    def map_m_step(self, data, weight, tau=1.0, prior=None):
        B, N, d = data.shape
        
        if prior is None:
            m, V = self.forward()
        else:
            m, V = prior
        
        wsum = weight.sum(1)
        wsum_reg = wsum + tau
        wxsum = torch.bmm(weight.permute(0,2,1), data)
        wxxsum = torch.bmm(weight.permute(0,2,1), data**2)
        pi = wsum_reg / wsum_reg.sum(1, keepdim=True)
        mu = (wxsum + m.unsqueeze(0)*tau) / wsum_reg.unsqueeze(-1)
        Sigma = (wxxsum + (V+m**2).unsqueeze(0)*tau) / wsum_reg.unsqueeze(-1) - mu**2

        return pi.float(), mu.float(), Sigma.float()
    
    def map_em(self, data, mask=None, num_iters=3, tau=1.0, prior=None):
        B, N, d = data.shape
        
        if mask is None:
            mask = torch.ones(B, N).to(data)

        # Need to set to the mode for initial starting point
        pi, mu, Sigma = self.mode(prior)
        pi = pi.unsqueeze(0).repeat(B,1)
        mu = mu.unsqueeze(0).repeat(B,1,1)
        Sigma = Sigma.unsqueeze(0).repeat(B,1,1)
        
        for emiter in range(num_iters):
            _, qq, _ = mog_eval((pi, mu, Sigma), data)
            qq = qq.exp() * mask.unsqueeze(-1)

            pi, mu, Sigma = self.map_m_step(data, weight=qq, tau=tau, prior=prior)
            
        return pi, mu, Sigma, qq

    def unsup_train(self, data_loader, n_samples_max=50000, use_cuda=False):
        """
        Find cluster centroids after spherical kmeans
        """

        if self.load_proto:
            print("Prototypes are already loaded")


class PANTHERBase(nn.Module):
    """
    Args:
    - p (int): Number of prototypes
    - d (int): Feature dimension
    - L (int): Number of EM iterations
    - out (str): Ways to merge features
    - ot_eps (float): eps
    """
    def __init__(self, d, p=5, L=3, tau=10.0, out='allcat', ot_eps=0.1,
                 load_proto=True, proto_path='.', fix_proto=True):
        super(PANTHERBase, self).__init__()

        self.L = L
        self.tau = tau
        self.out = out

        self.priors = DirNIWNet(p, d, ot_eps, load_proto, proto_path, fix_proto)

        if out == 'allcat':  # Concatenates pi, mu, cov
            self.outdim = p + 2*p*d
        elif out == 'weight_param_cat': # Concatenates mu and cov weighted by pi
            self.outdim = 2 * p * d
        elif 'select_top' in out:
            c = self.out[10:]
            numOfproto = 1 if c == '' else int(c)

            assert numOfproto <= p
            self.outdim = numOfproto * 2 * d + numOfproto
        elif 'select_bot' in out:
            c = self.out[10:]
            numOfproto = 1 if c == '' else int(c)

            assert numOfproto <= p
            self.outdim = numOfproto * 2 * d + numOfproto
        elif out == 'weight_avg_all':
            self.outdim = 2 * d
        elif out == 'weight_avg_mean':
            self.outdim = d
        else:
            raise NotImplementedError("Out mode {} not implemented".format(out))

    def forward(self, S, mask=None):
        """
        Args
        - S: data
        """
        B, N_max, d = S.shape
        
        if mask is None:
            mask = torch.ones(B, N_max).to(S)
        
        pis, mus, Sigmas, qqs = [], [], [], []
        pi, mu, Sigma, qq = self.priors.map_em(S, 
                                                    mask=mask, 
                                                    num_iters=self.L, 
                                                    tau=self.tau, 
                                                    prior=self.priors())

        pis.append(pi)
        mus.append(mu)
        Sigmas.append(Sigma)
        qqs.append(qq)

        pis = torch.stack(pis, dim=2) # pis: (n_batch x n_proto x n_head)
        mus = torch.stack(mus, dim=3) # mus: (n_batch x n_proto x instance_dim x n_head)
        Sigmas = torch.stack(Sigmas, dim=3) # Sigmas: (n_batch x n_proto x instance_dim x n_head)
        qqs = torch.stack(qqs, dim=3)

        if self.out == 'allcat':
            out = torch.cat([pis.reshape(B,-1),
                mus.reshape(B,-1), Sigmas.reshape(B,-1)], dim=1)
        elif self.out == 'weight_param_cat':
            out = []
            for h in range(self.H):
                pi, mu, Sigma = pis[..., h].reshape(B, -1), mus[..., h], Sigmas[..., h]
                mu_weighted = pi[..., None] * mu  # (n_batch, n_proto, instance_dim)
                Sigma_weighted = pi[..., None] * Sigma  # (n_batch, n_proto, instance_dim)

                out.append(mu_weighted.reshape(B, -1))
                out.append(Sigma_weighted.reshape(B, -1))

            out = torch.cat(out, dim=1)

        elif self.out == 'weight_avg_all':
            """
            Take weighted average of mu and sigmas according to estimated pi
            """
            out = []
            for h in range(self.H):
                pi, mu, Sigma = pis[..., h].reshape(B, 1, -1), mus[..., h], Sigmas[..., h]
                mu_weighted = torch.bmm(pi, mu).squeeze(dim=1)  # (n_batch, instance_dim)
                Sigma_weighted = torch.bmm(pi, Sigma).squeeze(dim=1)  # (n_batch, instance_dim)

                out.append(mu_weighted)
                out.append(Sigma_weighted)

            out = torch.cat(out, dim=1)

        elif self.out == 'weight_avg_mean':
            """
            Take weighted average of mu according to estimated pi
            """
            out = []
            for h in range(self.H):
                pi, mu = pis[..., h].reshape(B, 1, -1), mus[..., h]
                mu_weighted = torch.bmm(pi, mu).squeeze(dim=1)  # (n_batch, instance_dim)

                out.append(mu_weighted)

            out = torch.cat(out, dim=1)

        elif 'select_top' in self.out:
            c = self.out[10:]
            numOfproto = 1 if c == '' else int(c)

            out = []
            for h in range(self.H):
                pi, mu, Sigma = pis[..., h], mus[..., h], Sigmas[..., h]
                _, indices = torch.topk(pi, numOfproto, dim=1)

                out.append(pi[:, indices].reshape(pi.shape[0], -1))
                out.append(mu[:, indices].reshape(mu.shape[0], -1))
                out.append(Sigma[:, indices].reshape(Sigma.shape[0], -1))
            out = torch.cat(out, dim=1)

        elif 'select_bot' in self.out:
            c = self.out[10:]
            numOfproto = 1 if c == '' else int(c)

            out = []
            for h in range(self.H):
                pi, mu, Sigma = pis[..., h], mus[..., h], Sigmas[..., h]
                _, indices = torch.topk(-pi, numOfproto, dim=1)

                out.append(pi[:, indices].reshape(pi.shape[0], -1))
                out.append(mu[:, indices].reshape(mu.shape[0], -1))
                out.append(Sigma[:, indices].reshape(Sigma.shape[0], -1))
            out = torch.cat(out, dim=1)

        else:
            raise NotImplementedError

        return out, qqs
    

class DiffEMModule(MIMIC4LightningModule):
    def __init__(self,
                 task: str = "ihm",
                 modeltype: str = "TS",
                 max_epochs: int = 10,
                 img_learning_rate: float = 1e-4,
                 ts_learning_rate: float = 4e-4,
                 period_length: int = 48,
                 num_prototypes: int = 10,
                 orig_reg_d_ts: int = 30,
                 hidden_dim: int = 128,
                 n_layers: int = 3,
                 *args,
                 **kwargs):
        super().__init__(task=task, modeltype=modeltype, max_epochs=max_epochs,
                         img_learning_rate=img_learning_rate, ts_learning_rate=ts_learning_rate,
                         period_length=period_length)

        self.input_size = orig_reg_d_ts
        self.hidden_dim = hidden_dim
        self.num_prototypes = num_prototypes

        # self.feature_extractor = InceptionTimeFeatureExtractor(orig_reg_d_ts, out_channels=hidden_dim // 4)
        self.feature_extractor = ResNetFeatureExtractor(n_in_channels=orig_reg_d_ts, 
                                                        out_channels=self.hidden_dim)
        self.panther = PANTHERBase(self.hidden_dim, p=config.out_size, L=config.em_iter,
                                  tau=config.tau, out=config.out_type, ot_eps=config.ot_eps,
                                  load_proto=config.load_proto, proto_path=config.proto_path,
                                  fix_proto=config.fix_proto)
        dropout = 0.1
        apply_positional_encoding = True
        self.pool = MILConjunctivePooling(
            self.hidden_dim,
            self.num_labels,
            dropout=dropout,
            apply_positional_encoding=apply_positional_encoding
        )

    def forward(self,
                reg_ts,
                labels=None,
                **kwargs):
        
        batch_size = reg_ts.size(0)
        x = reg_ts.permute(0, 2, 1)
        feat = self.feature_extractor(x)
        # add OTK layer
        att_feat = self.ot_layer(feat.permute(0, 2, 1))
        pool_output = self.pool(att_feat.permute(0, 2, 1))
        output = pool_output["bag_logits"]

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
    # # test OTkernel
    # ot_layer = OTKernel(30, 10, heads=1, eps=1.0, max_iter=30)
    # input = torch.randn(4, 48, 30)
    # out = ot_layer(input)
    # print(out.shape)
    # ipdb.set_trace()
    from cmehr.paths import *
    from cmehr.dataset.mimic4_pretraining_datamodule import MIMIC4DataModule

    datamodule = MIMIC4DataModule(
        file_path=str(ROOT_PATH / "output_mimic4/TS_CXR/ihm"),
        modeltype="TS",
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
    model = DiffEMModule(
    )
    loss = model(
        reg_ts=batch["reg_ts"],
        labels=batch["label"]
    )
    print(loss)