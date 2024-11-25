import numpy as np
from PIL import Image
import math
# import ot
import ipdb
from einops import rearrange
from typing import Optional, List
from torch import nn
import torch.nn.functional as F
import torch
from cmehr.models.mimic4.UTDE_modules import multiTimeAttention, gateMLP
from cmehr.models.mimic4.base_model import MIMIC4LightningModule
from cmehr.models.mimic4.position_encode import PositionalEncoding1D

# class OT_Attn_assem(nn.Module):
#     def __init__(self, impl='pot-uot-l2', ot_reg=0.1, ot_tau=0.5) -> None:
#         super().__init__()
#         self.impl = impl
#         self.ot_reg = ot_reg
#         self.ot_tau = ot_tau
#         print("ot impl: ", impl)

#     def normalize_feature(self, x):
#         x = x - x.min(-1)[0].unsqueeze(-1)
#         return x

#     def OT(self, weight1, weight2):
#         """
#         Parmas:
#             weight1 : (N, D)
#             weight2 : (M, D)

#         Return:
#             flow : (N, M)
#             dist : (1, )
#         """

#         if self.impl == "pot-sinkhorn-l2":
#             self.cost_map = torch.cdist(weight1, weight2)**2  # (N, M)

#             src_weight = weight1.sum(dim=1) / weight1.sum()
#             dst_weight = weight2.sum(dim=1) / weight2.sum()

#             cost_map_detach = self.cost_map.detach()
#             flow = ot.sinkhorn(a=src_weight.detach(), b=dst_weight.detach(),
#                                M=cost_map_detach/cost_map_detach.max(), reg=self.ot_reg)
#             dist = self.cost_map * flow
#             dist = torch.sum(dist)
#             return flow, dist

#         elif self.impl == "pot-uot-l2":
#             a, b = ot.unif(weight1.size()[0]).astype(
#                 'float64'), ot.unif(weight2.size()[0]).astype('float64')
#             self.cost_map = torch.cdist(weight1, weight2)**2  # (N, M)

#             cost_map_detach = self.cost_map
#             M_cost = cost_map_detach / cost_map_detach.max()

#             flow = ot.unbalanced.sinkhorn_knopp_unbalanced(a=a, b=b,
#                                                            M=M_cost.detach().double().cpu().numpy(),
#                                                            reg=self.ot_reg,
#                                                            reg_m=self.ot_tau)
#             flow = torch.from_numpy(flow).type_as(weight1)
#             dist = self.cost_map * flow  # (N, M)
#             dist = torch.sum(dist)  # (1,) float
#             return flow, dist

#         else:
#             raise NotImplementedError

#     def forward(self, x, y):
#         '''
#         x: (N, D)
#         y: (M, D)
#         '''
#         x = self.normalize_feature(x)
#         y = self.normalize_feature(y)
#         pi, dist = self.OT(x, y)

#         return pi, dist


# def compute_optimal_transport(M, r, c, epsilon=1e-6, lam=10):
#     n_runs, n, m = M.shape
#     P = torch.exp(-lam * M)
#     P /= P.view((n_runs, -1)).sum(1).unsqueeze(1).unsqueeze(1)

#     u = torch.zeros(n_runs, n)
#     maxiters = 1000
#     iters = 1
#     # normalize this matrix
#     while torch.max(torch.abs(u - P.sum(2))) > epsilon:
#         u = P.sum(2)
#         P *= (r / u).view((n_runs, -1, 1))
#         P *= (c / P.sum(1)).view((n_runs, 1, -1))
#         if iters == maxiters:
#             break
#         iters = iters + 1

#     return P, (P.detach() * M).sum(-1).sum(-1)


# class SemanticGrouping(nn.Module):
#     def __init__(self, num_slots, dim_slot, temp=0.07, eps=1e-6):
#         super().__init__()
#         self.num_slots = num_slots
#         self.dim_slot = dim_slot
#         self.temp = temp
#         self.eps = eps

#         self.slot_embed = nn.Embedding(num_slots, dim_slot)

#     def forward(self, x):
#         x_prev = x
#         slots = self.slot_embed(torch.arange(
#             0, self.num_slots, device=x.device)).unsqueeze(0).repeat(x.size(0), 1, 1)
#         dots = torch.einsum(
#             'bkd,bnd->bkn', F.normalize(slots, dim=-1), F.normalize(x, dim=-1))
#         attn = (dots / self.temp).softmax(dim=1) + self.eps
#         slots = torch.einsum('bnd,bkn->bkd', x_prev,
#                              attn / attn.sum(dim=2, keepdim=True))
#         return slots, dots


# class ScouterAttention(nn.Module):
#     def __init__(self, dim, num_concept, iters=3, eps=1e-8, vis=False, power=1, to_k_layer=3):
#         super().__init__()

#         self.num_slots = num_concept
#         self.iters = iters
#         self.eps = eps
#         # self.scale = (dim // num_concept) ** -0.5
#         self.scale = dim ** -0.5

#         # random seed init
#         slots_mu = nn.Parameter(torch.randn(1, 1, dim))
#         slots_sigma = nn.Parameter(torch.abs(torch.randn(1, 1, dim)))
#         mu = slots_mu.expand(1, self.num_slots, -1)
#         sigma = slots_sigma.expand(1, self.num_slots, -1)
#         self.initial_slots = nn.Parameter(torch.normal(mu, sigma))

#         # K layer init
#         to_k_layer_list = [nn.Linear(dim, dim)]
#         for to_k_layer_id in range(1, to_k_layer):
#             to_k_layer_list.append(nn.ReLU(inplace=True))
#             to_k_layer_list.append(nn.Linear(dim, dim))
#         self.to_k = nn.Sequential(
#             *to_k_layer_list
#         )

#         self.vis = vis
#         self.power = power

#     def forward(self, inputs_pe, inputs, weight=None, things=None):
#         b, n, d = inputs_pe.shape
#         slots = self.initial_slots.expand(b, -1, -1)
#         k, v = self.to_k(inputs_pe), inputs_pe
#         for _ in range(self.iters):
#             q = slots

#             dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
#             dots = torch.div(dots, dots.sum(2).expand_as(dots.permute([2, 0, 1])).permute([1, 2, 0])) * \
#                 dots.sum(2).sum(1).expand_as(
#                     dots.permute([1, 2, 0])).permute([2, 0, 1])
#             attn = torch.sigmoid(dots)

#             attn2 = attn / (attn.sum(dim=-1, keepdim=True) + self.eps)
#             updates = torch.einsum('bjd,bij->bid', inputs, attn2)

#         if self.vis:
#             slots_vis_raw = attn.clone()
#             vis(slots_vis_raw, "vis", self.args.feature_size, weight, things)

#         return updates, attn


class SlotAttention(nn.Module):
    '''
    Implementation of original slot attention.
    '''

    def __init__(self, num_slots, dim, iters=3, eps=1e-8, hidden_dim=128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_sigma = nn.Parameter(torch.rand(1, 1, dim))

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs, num_slots=None):
        b, n, d = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots
        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_sigma.expand(b, n_s, -1)
        slots = torch.normal(mu, sigma)

        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum('bjd,bij->bid', v, attn)

            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )

            slots = slots.reshape(b, -1, d)
            slots = slots + self.fc2(F.relu(self.fc1(self.norm_pre_ff(slots))))

        return slots, attn

# class CrossAttention(nn.Module):
#     def __init__(
#         self, dim, n_outputs=None, num_heads=8, attention_dropout=0.1, projection_dropout=0.0
#     ):
#         super().__init__()
#         n_outputs = n_outputs if n_outputs else dim
#         self.num_heads = num_heads
#         head_dim = dim // self.num_heads
#         self.scale = head_dim ** -0.5

#         self.q = nn.Linear(dim, dim, bias=False)
#         self.kv = nn.Linear(dim, dim * 2, bias=False)
#         self.attn_drop = nn.Dropout(attention_dropout)

#         self.proj = nn.Linear(dim, n_outputs)
#         self.proj_drop = nn.Dropout(projection_dropout)

#     def forward(self, x, y):
#         B, Nx, C = x.shape
#         By, Ny, Cy = y.shape

#         assert C == Cy, "Feature size of x and y must be the same"

#         q = self.q(x).reshape(B, Nx, 1, self.num_heads, C //
#                               self.num_heads).permute(2, 0, 3, 1, 4)
#         kv = (
#             self.kv(y)
#             .reshape(By, Ny, 2, self.num_heads, C // self.num_heads)
#             .permute(2, 0, 3, 1, 4)
#         )

#         q = q[0]
#         k, v = kv[0], kv[1]

#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)

#         x = (attn @ v).transpose(1, 2).reshape(B, Nx, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x, attn


class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)  # 4, 50, 64
        A = self.attention_c(A)  # N x n_classes
        return A, x


# class SlotAttention(nn.Module):
#     def __init__(self, num_classes, slots_per_class, dim,
#                  iters=3, eps=1e-8, vis=False, vis_id=0,
#                  loss_status=1, power=1, to_k_layer=1):
#         super().__init__()
#         self.num_classes = num_classes
#         self.slots_per_class = slots_per_class
#         self.num_slots = num_classes * slots_per_class
#         self.iters = iters
#         self.eps = eps
#         self.scale = dim ** -0.5
#         self.loss_status = loss_status

#         slots_mu = nn.Parameter(torch.randn(1, 1, dim))
#         slots_sigma = nn.Parameter(torch.abs(torch.randn(1, 1, dim)))

#         mu = slots_mu.expand(1, self.num_slots, -1)
#         sigma = slots_sigma.expand(1, self.num_slots, -1)
#         self.initial_slots = nn.Parameter(torch.normal(mu, sigma))

#         self.to_q = nn.Sequential(
#             nn.Linear(dim, dim),
#         )
#         to_k_layer_list = [nn.Linear(dim, dim)]
#         for to_k_layer_id in range(1, to_k_layer):
#             to_k_layer_list.append(nn.ReLU(inplace=True))
#             to_k_layer_list.append(nn.Linear(dim, dim))

#         self.to_k = nn.Sequential(
#             *to_k_layer_list
#         )
#         self.gru = nn.GRU(dim, dim)

#         self.vis = vis
#         self.vis_id = vis_id
#         self.power = power

#     def forward(self, inputs, inputs_x):
#         b, n, d = inputs.shape
#         slots = self.initial_slots.expand(b, -1, -1)
#         k, v = self.to_k(inputs), inputs

#         for _ in range(self.iters):
#             slots_prev = slots

#             # q = self.to_q(slots)
#             q = slots

#             dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
#             dots = torch.div(dots, dots.sum(2).expand_as(dots.permute([2, 0, 1])).permute(
#                 [1, 2, 0])) * dots.sum(2).sum(1).expand_as(dots.permute([1, 2, 0])).permute([2, 0, 1])  # * 10
#             attn = torch.sigmoid(dots)
#             updates = torch.einsum('bjd,bij->bid', inputs_x, attn)
#             updates = updates / inputs_x.size(2)
#             self.gru.flatten_parameters()
#             slots, _ = self.gru(
#                 updates.reshape(1, -1, d),
#                 slots_prev.reshape(1, -1, d)
#             )

#             slots = slots.reshape(b, -1, d)

#             if self.vis:
#                 slots_vis = attn.clone()

#         if self.vis:
#             if self.slots_per_class > 1:
#                 new_slots_vis = torch.zeros(
#                     (slots_vis.size(0), self.num_classes, slots_vis.size(-1)))
#                 for slot_class in range(self.num_classes):
#                     new_slots_vis[:, slot_class] = torch.sum(torch.cat(
#                         [slots_vis[:, self.slots_per_class*slot_class: self.slots_per_class*(slot_class+1)]], dim=1), dim=1, keepdim=False)
#                 slots_vis = new_slots_vis.to(updates.device)

#             slots_vis = slots_vis[self.vis_id]
#             slots_vis = ((slots_vis - slots_vis.min()) / (slots_vis.max()-slots_vis.min()) * 255.).reshape(
#                 slots_vis.shape[:1]+(int(slots_vis.size(1)**0.5), int(slots_vis.size(1)**0.5)))
#             slots_vis = (slots_vis.cpu().detach().numpy()).astype(np.uint8)
#             for id, image in enumerate(slots_vis):
#                 image = Image.fromarray(image, mode='L')
#                 image.save(f'sloter/vis/slot_{id:d}.png')
#             print(self.loss_status*torch.sum(attn.clone(), dim=2, keepdim=False))
#             print(self.loss_status*torch.sum(updates.clone(), dim=2, keepdim=False))

#         if self.slots_per_class > 1:
#             new_updates = torch.zeros(
#                 (updates.size(0), self.num_classes, updates.size(-1)))
#             for slot_class in range(self.num_classes):
#                 new_updates[:, slot_class] = torch.sum(
#                     updates[:, self.slots_per_class*slot_class: self.slots_per_class*(slot_class+1)], dim=1, keepdim=False)
#             updates = new_updates.to(updates.device)

#         attn_relu = torch.relu(attn)
#         slot_loss = torch.sum(attn_relu) / attn.size(0) / \
#             attn.size(1) / attn.size(2)  # * self.slots_per_class

#         return self.loss_status*torch.sum(updates, dim=2, keepdim=False), torch.pow(slot_loss, self.power)


class ProtoTSModel(MIMIC4LightningModule):
    def __init__(self,
                 task: str = "ihm",
                 modeltype: str = "TS",
                 max_epochs: int = 10,
                 img_learning_rate: float = 1e-4,
                 ts_learning_rate: float = 4e-4,
                 period_length: int = 48,
                 use_prototype: bool = True,
                 use_multiscale: bool = True,
                 orig_d_ts: int = 15,
                 orig_reg_d_ts: int = 30,
                 num_heads: int = 8,
                 layers: int = 3,
                 dropout: float = 0.1,
                 irregular_learn_emb_ts: bool = True,
                 reg_ts: bool = True,
                 TS_mixup: bool = True,
                 mixup_level: str = "batch",
                 pooling_type: str = "attention",
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

        self.num_heads = num_heads
        self.layers = layers
        self.dropout = dropout
        self.irregular_learn_emb_ts = irregular_learn_emb_ts
        self.reg_ts = reg_ts
        self.TS_mixup = TS_mixup
        self.mixup_level = mixup_level
        self.task = task
        self.tt_max = period_length
        self.orig_d_ts = orig_d_ts
        self.d_ts = embed_dim
        self.use_multiscale = use_multiscale
        self.use_prototype = use_prototype
        self.pooling_type = pooling_type
        self.lamb = lamb

        if self.irregular_learn_emb_ts:
            # formulate the regular time stamps
            self.periodic = nn.Linear(1, embed_time-1)
            self.linear = nn.Linear(1, 1)
            self.time_query = torch.linspace(0, 1., self.tt_max)
            self.time_attn_ts = multiTimeAttention(
                self.orig_d_ts*2, self.d_ts, embed_time, 8)

        if self.reg_ts:
            self.orig_reg_d_ts = orig_reg_d_ts
            self.proj_ts = nn.Conv1d(self.orig_reg_d_ts, self.d_ts, kernel_size=1, padding=0,
                                     bias=False)

        if self.TS_mixup:
            if self.mixup_level == 'batch':
                self.moe = gateMLP(
                    input_dim=self.d_ts*2, hidden_size=embed_dim, output_dim=1, dropout=dropout)
            elif self.mixup_level == 'batch_seq':
                self.moe = gateMLP(
                    input_dim=self.d_ts*2, hidden_size=embed_dim, output_dim=1, dropout=dropout)
            elif self.mixup_level == 'batch_seq_feature':
                self.moe = gateMLP(
                    input_dim=self.d_ts*2, hidden_size=embed_dim, output_dim=self.d_ts, dropout=dropout)
            else:
                raise ValueError("Unknown mixedup type")

        self.proj_multi_scale_ts = nn.ModuleList()
        if self.use_multiscale:
            for step in [1, 4, 8]:
                proj_ts = nn.Conv1d(self.d_ts, self.d_ts,
                                    kernel_size=step, stride=step,
                                    bias=False)
                self.proj_multi_scale_ts.append(proj_ts)
        else:
            proj_ts = nn.Conv1d(self.d_ts, self.d_ts,
                                kernel_size=1, stride=1,
                                bias=False)
            self.proj_multi_scale_ts.append(proj_ts)

        self.proj1 = nn.Linear(self.d_ts, self.d_ts)
        self.proj2 = nn.Linear(self.d_ts, self.d_ts)
        self.out_layer = nn.Linear(self.d_ts, self.num_labels)

        if self.use_prototype:
            self.pe = PositionalEncoding1D(embed_dim)
            if self.use_multiscale:
                num_prototypes = [20, 10, 5]
            else:
                num_prototypes = [10]
            self.grouping = []
            for i, num_prototype in enumerate(num_prototypes):
                self.grouping.append(SlotAttention(
                    dim=embed_dim, num_slots=num_prototype))
            self.grouping = nn.ModuleList(self.grouping)

        self.atten_pooling = Attn_Net_Gated(
            L=embed_dim, D=64, dropout=True, n_classes=1)

    def learn_time_embedding(self, tt):
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)

    def forward_ts_mtand(self,
                         x_ts: torch.Tensor,
                         x_ts_mask: torch.Tensor,
                         ts_tt_list: torch.Tensor):
        '''
        Forward irregular time series using mTAND.
        '''
        if self.irregular_learn_emb_ts:
            # (B, N) -> (B, N, embed_time)
            # fixed
            time_key_ts = self.learn_time_embedding(
                ts_tt_list)
            x_ts_irg = torch.cat((x_ts, x_ts_mask), 2)
            x_ts_mask = torch.cat((x_ts_mask, x_ts_mask), 2)

            time_query = self.learn_time_embedding(
                self.time_query.unsqueeze(0).type_as(x_ts))
            # query: (1, N_r, embed_time),
            # key: (B, N, embed_time),
            # value: (B, N, 2 * D_t)
            # mask: (B, N, 2 * D_t)
            # out: (B, N_r, 128?)
            proj_x_ts_irg = self.time_attn_ts(
                time_query, time_key_ts, x_ts_irg, x_ts_mask)
            proj_x_ts_irg = proj_x_ts_irg.transpose(0, 1)
        else:
            raise ValueError("Not implemented")

        return proj_x_ts_irg

    def forward_ts_reg(self, reg_ts: torch.Tensor):
        '''
        Forward irregular time series using Imputation.
        '''
        # convolution over regular time series

        x_ts_reg = reg_ts.transpose(1, 2)
        proj_x_ts_reg = x_ts_reg if self.orig_reg_d_ts == self.d_ts else self.proj_ts(
            x_ts_reg)
        proj_x_ts_reg = proj_x_ts_reg.permute(2, 0, 1)

        return proj_x_ts_reg

    def gate_ts(self,
                proj_x_ts_irg: torch.Tensor,
                proj_x_ts_reg: torch.Tensor):

        if self.TS_mixup:
            if self.mixup_level == 'batch':
                g_irg = torch.max(proj_x_ts_irg, dim=0).values
                g_reg = torch.max(proj_x_ts_reg, dim=0).values
                moe_gate = torch.cat([g_irg, g_reg], dim=-1)
            elif self.mixup_level == 'batch_seq' or self.mixup_level == 'batch_seq_feature':
                moe_gate = torch.cat(
                    [proj_x_ts_irg, proj_x_ts_reg], dim=-1)
            else:
                raise ValueError("Unknown mixup type")
            mixup_rate = self.moe(moe_gate)
            proj_x_ts = mixup_rate * proj_x_ts_irg + \
                (1 - mixup_rate) * proj_x_ts_reg
        else:
            if self.irregular_learn_emb_ts:
                proj_x_ts = proj_x_ts_irg
            elif self.reg_ts:
                proj_x_ts = proj_x_ts_reg
            else:
                raise ValueError("Unknown time series type")

        return proj_x_ts

    def forward(self, x_ts, x_ts_mask, ts_tt_list,
                cxr_imgs_sequences=None,
                cxr_time_sequences=None,
                cxr_time_mask_sequence=None,
                labels=None, reg_ts=None):
        """ Forward function of Multimodal model
        :param x_ts: (B, N, D_t), torch.Tensor, time series data
        :param x_ts_mask: (B, N, D_t), torch.Tensor, time series mask
        :param ts_tt_list: (B, N), torch.Tensor, time series time
        :param labels: (B, ), torch.Tensor, labels
        :param reg_ts: (B, N_r, D_r), torch.Tensor, regular time series data
        """

        proj_x_ts_irg = self.forward_ts_mtand(
            x_ts, x_ts_mask, ts_tt_list)
        assert reg_ts != None
        proj_x_ts_reg = self.forward_ts_reg(reg_ts)
        # 42, 4, 128
        proj_x_ts = self.gate_ts(proj_x_ts_irg, proj_x_ts_reg)
        batch_size = proj_x_ts.size(0)
        proj_x_ts = rearrange(proj_x_ts, "tt b d -> b d tt")
        slot_loss = 0.

        if not self.use_multiscale:
            ts_feat = self.proj_multi_scale_ts[0](proj_x_ts)
            if self.use_prototype:
                ts_feat = rearrange(ts_feat, "b d tt -> b tt d")
                pe = self.pe(ts_feat)
                ts_pe = ts_feat + pe
                updates, attn = self.grouping[0](ts_pe)
                slot_loss += torch.mean(attn)
                last_ts_feat = torch.cat([updates, ts_feat], dim=1)
            else:
                last_ts_feat = rearrange(ts_feat, "b d tt -> b tt d")
        else:
            # proj_x_ts = rearrange(proj_x_ts, "tt b d -> b d tt")
            multi_scale_feats = []
            if self.use_prototype:
                for idx, proj_ts in enumerate(self.proj_multi_scale_ts):
                    # extract the feature in each window
                    ts_feat = proj_ts(proj_x_ts)
                    ts_feat = rearrange(ts_feat, "b d tt -> b tt d")
                    pe = self.pe(ts_feat)
                    ts_pe = ts_feat + pe
                    updates, attn = self.grouping[idx](ts_pe)
                    slot_loss += torch.mean(attn)
                    multi_scale_feats.append(updates)
                    multi_scale_feats.append(ts_feat)

                slot_loss /= len(self.proj_multi_scale_ts)

                last_ts_feat = torch.cat(multi_scale_feats, dim=1)
                # concat patterns with timestamp features
                # last_ts_feat = torch.cat(
                #     [multiscale_feats, rearrange(proj_x_ts, "b d tt -> b tt d")], dim=1)
            else:
                for proj_ts in self.proj_multi_scale_ts:
                    ts_feat = proj_ts(proj_x_ts)
                    multi_scale_feats.append(
                        rearrange(ts_feat, "b d tt -> b tt d"))
                last_ts_feat = torch.cat(multi_scale_feats, dim=1)

        '''
        TODO: Maybe some regularization are needed
        '''
        # attention pooling
        if self.pooling_type == "attention":
            attn, last_ts_feat = self.atten_pooling(last_ts_feat)
            last_hs = torch.bmm(attn.permute(0, 2, 1),
                                last_ts_feat).squeeze(dim=1)
        elif self.pooling_type == "mean":
            last_hs = last_ts_feat.mean(dim=1)
        elif self.pooling_type == "last":
            last_hs = last_ts_feat[:, -1, :]

        # MLP for the final prediction
        last_hs_proj = self.proj2(
            F.dropout(F.relu(self.proj1(last_hs)), p=self.dropout, training=self.training))
        last_hs_proj += last_hs
        output = self.out_layer(last_hs_proj)

        if self.task in ['ihm', 'readm']:
            if labels != None:
                ce_loss = self.loss_fct1(output, labels)
                return ce_loss + self.lamb * slot_loss
            return F.softmax(output, dim=-1)[:, 1]

        elif self.task == 'pheno':
            if labels != None:
                labels = labels.float()
                ce_loss = self.loss_fct1(output, labels)
                return ce_loss + self.lamb * slot_loss
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
    model = ProtoTSModel(
        use_multiscale=True,
        use_prototype=True
    )
    loss = model(
        x_ts=batch["ts"],  # type ignore
        x_ts_mask=batch["ts_mask"],
        ts_tt_list=batch["ts_tt"],
        reg_ts=batch["reg_ts"],
        labels=batch["label"],
    )
    print(loss)

    # feat1 = torch.randn(12, 128)
    # feat2 = torch.randn(48, 128)
    # coattn = OT_Attn_assem(impl="pot-uot-l2", ot_reg=0.05, ot_tau=0.5)
    # A_coattn, dist = coattn(feat1, feat2)
    # A_coattn: optimal transport matrix feat1 -> feat2
