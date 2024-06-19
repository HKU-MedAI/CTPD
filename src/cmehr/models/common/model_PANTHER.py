# Model initiation for PANTHER
import os
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from cmehr.utils.file_utils import load_pkl
import ipdb


class PrototypeTokenizer(nn.Module):
    """
    Tokenize the prototype features, so that we have access to mixture probabilities/mean/covariance
    """
    def __init__(self, proto_model_type='PANTHER', out_type='allcat', p=8):
        super().__init__()
        self.model_type = proto_model_type
        self.p = p
        self.out_type = out_type

    def get_eff_dim(self):
        return 2 * self.p

    def forward(self, X):
        n_samples = X.shape[0]
        if self.model_type == 'OT':
            if self.out_type == 'allcat':
                prob = 1 / self.p * torch.ones((n_samples, self.p)).type_as(X)
                mean = X.reshape(n_samples, self.p, -1)
                cov = None
            else:
                raise NotImplementedError(f"Not implemented for {self.out_type}")

        elif self.model_type == 'PANTHER':
            if self.out_type == 'allcat':
                d = (X.shape[1] - self.p) // (2 * self.p)
                prob = X[:, : self.p]
                mean = X[:, self.p: self.p * (1 + d)].reshape(-1, self.p, d)
                cov = X[:, self.p * (1 + d):].reshape(-1, self.p, d)
            else:
                raise NotImplementedError(f"Not implemented for {self.out_type}")
        else:
            raise NotImplementedError(f"Not implemented for {self.model_type}")


        return prob, mean, cov
    

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
    

def predict_emb(self, dataloader, use_cuda=True, permute=False):
    """
    Create prototype-based slide representation

    Returns
    - X (torch.Tensor): (n_data x output_set_dim)
    - y (torch.Tensor): (n_data)
    """

    X = []
    y = []
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        if use_cuda:
            emb = batch[0].cuda()
        
        with torch.no_grad():
            out = self.representation(emb)
            out = out['repr'].data.detach().cpu()

        X.append(out)
        y.append(batch[1])

    X = torch.cat(X)
    y = torch.cat(y)

    return X, y


def check_prototypes(n_proto, embed_dim, load_proto, proto_path):
    """
    Check validity of the prototypes
    """
    if load_proto:
        assert os.path.exists(proto_path), "{} does not exist!".format(proto_path)
        if proto_path.endswith('pkl'):
            prototypes = load_pkl(proto_path)['prototypes'].squeeze()
        elif proto_path.endswith('npy'):
            prototypes = np.load(proto_path)


        assert (n_proto == prototypes.shape[0]) and (embed_dim == prototypes.shape[1]),\
            "Prototype dimensions do not match! Params: ({}, {}) Suplied: ({}, {})".format(n_proto,
                                                                                           embed_dim,
                                                                                           prototypes.shape[0],
                                                                                           prototypes.shape[1])


class PANTHER(nn.Module):
    """
    Wrapper for PANTHER model
    """
    def __init__(self, 
                 in_dim: int = 128, 
                 heads: int = 1, 
                 out_size: int = 50, 
                 load_proto: bool = True, 
                 em_iter: int = 1, 
                 tau: float = 1.0,
                 out_type: str = "allcat", 
                 ot_eps: float = 1, 
                 proto_path: str = ".", 
                 fix_proto: bool = True, 
                 mode: str = "emb"):
        
        super(PANTHER, self).__init__()

        emb_dim = in_dim

        self.emb_dim = emb_dim
        self.heads = heads
        self.outsize = out_size
        self.load_proto = load_proto
        self.mode = mode

        # FIXME: maybe we need to check prototype file?
        check_prototypes(out_size, self.emb_dim, self.load_proto, proto_path)
        # This module contains the EM step
        self.panther = PANTHERBase(self.emb_dim, p=out_size, L=em_iter,
                                   tau=tau, out=out_type, ot_eps=ot_eps,
                                   load_proto=load_proto, proto_path=proto_path,
                                   fix_proto=fix_proto)

    def representation(self, x):
        """
        Construct unsupervised slide representation
        """
        out, qqs = self.panther(x)
        return {'repr': out, 'qq': qqs}

    def forward(self, x):
        out = self.representation(x)
        return out['repr']
    
    def predict(self, data_loader, use_cuda=True):
        if self.mode == 'classification':
            pass
        elif self.mode == 'survival':
            pass
        elif self.mode == 'emb':
            output, y = predict_emb(self, data_loader, use_cuda=use_cuda)
        else:
            raise NotImplementedError(f"Not implemented for {self.mode}!")
        
        return output, y
