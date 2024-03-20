import math
import ot
import ipdb
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from .UTDE_modules import multiTimeAttention, gateMLP, BertForRepresentation, \
    MAGGate, Outer, TransformerEncoder, TransformerCrossEncoder, TimeSeriesCnnModel


class OT_Attn_assem(nn.Module):
    def __init__(self, impl='pot-uot-l2', ot_reg=0.1, ot_tau=0.5) -> None:
        super().__init__()
        self.impl = impl
        self.ot_reg = ot_reg
        self.ot_tau = ot_tau
        print("ot impl: ", impl)

    def normalize_feature(self, x):
        x = x - x.min(-1)[0].unsqueeze(-1)
        return x

    def OT(self, weight1, weight2):
        """
        Parmas:
            weight1 : (N, D)
            weight2 : (M, D)

        Return:
            flow : (N, M)
            dist : (1, )
        """

        if self.impl == "pot-sinkhorn-l2":
            self.cost_map = torch.cdist(weight1, weight2)**2  # (N, M)

            src_weight = weight1.sum(dim=1) / weight1.sum()
            dst_weight = weight2.sum(dim=1) / weight2.sum()

            cost_map_detach = self.cost_map.detach()
            flow = ot.sinkhorn(a=src_weight.detach(), b=dst_weight.detach(),
                               M=cost_map_detach/cost_map_detach.max(), reg=self.ot_reg)
            dist = self.cost_map * flow
            dist = torch.sum(dist)
            return flow, dist

        elif self.impl == "pot-uot-l2":
            a, b = ot.unif(weight1.size()[0]).astype(
                'float64'), ot.unif(weight2.size()[0]).astype('float64')
            self.cost_map = torch.cdist(weight1, weight2)**2  # (N, M)

            cost_map_detach = self.cost_map.detach()
            M_cost = cost_map_detach/cost_map_detach.max()

            flow = ot.unbalanced.sinkhorn_knopp_unbalanced(a=a, b=b,
                                                           M=M_cost.double().cpu().numpy(), reg=self.ot_reg, reg_m=self.ot_tau)
            flow = torch.from_numpy(flow).type_as(weight1)
            dist = self.cost_map * flow  # (N, M)
            dist = torch.sum(dist)  # (1,) float
            return flow, dist

        else:
            raise NotImplementedError

    def forward(self, x, y):
        '''
        x: (N, D)
        y: (M, D)
        '''
        x = self.normalize_feature(x)
        y = self.normalize_feature(y)
        pi, dist = self.OT(x, y)

        return pi, dist


class CrossAttention(nn.Module):
    def __init__(
        self, dim, n_outputs=None, num_heads=8, attention_dropout=0.1, projection_dropout=0.0
    ):
        super().__init__()
        n_outputs = n_outputs if n_outputs else dim
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, dim * 2, bias=False)
        self.attn_drop = nn.Dropout(attention_dropout)

        self.proj = nn.Linear(dim, n_outputs)
        self.proj_drop = nn.Dropout(projection_dropout)

    def forward(self, x, y):
        B, Nx, C = x.shape
        By, Ny, Cy = y.shape

        assert C == Cy, "Feature size of x and y must be the same"

        q = self.q(x).reshape(B, Nx, 1, self.num_heads, C //
                              self.num_heads).permute(2, 0, 3, 1, 4)
        kv = (
            self.kv(y)
            .reshape(By, Ny, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )

        q = q[0]
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, Nx, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class ProtoMULT(nn.Module):
    def __init__(self,
                 task: str = "ihm",
                 orig_d_ts: int = 17,
                 orig_reg_d_ts: int = 34,
                 orig_d_txt: int = 768,
                 tt_max: int = 48,
                 num_of_notes: int = 5,
                 num_heads: int = 8,
                 layers: int = 3,
                 kernel_size: int = 1,
                 dropout: float = 0.1,
                 irregular_learn_emb_ts: bool = True,
                 irregular_learn_emb_text: bool = True,
                 reg_ts: bool = True,
                 TS_mixup: bool = True,
                 mixup_level: str = "batch",
                 cross_method: str = "self_cross",
                 embed_time: int = 64,
                 embed_dim: int = 128,
                 model_name: str = "yikuan8/Clinical-Longformer",
                 num_labels: int = 2,
                 cross_layers: int = 3,
                 ):
        """
        Construct a MulT Cross model.
        :task: str, the task of the model, e.g. "ihm", "pheno", "decomp", "los"
        :orig_d_ts: int, the original dimension of the time series data
        :orig_reg_d_ts: int, the original dimension of the regular time series data
        """
        super(ProtoMULT, self).__init__()

        ts_seq_num = tt_max
        text_seq_num = num_of_notes

        self.num_heads = num_heads
        self.layers = layers
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.attn_mask = False
        self.irregular_learn_emb_ts = irregular_learn_emb_ts
        self.irregular_learn_emb_text = irregular_learn_emb_text
        self.reg_ts = reg_ts
        self.TS_mixup = TS_mixup
        self.mixup_level = mixup_level
        self.task = task
        self.tt_max = tt_max
        self.cross_method = cross_method

        if self.irregular_learn_emb_ts or self.irregular_learn_emb_text:
            # formulate the regular time stamps
            self.time_query = torch.linspace(0, 1., self.tt_max)
            self.periodic = nn.Linear(1, embed_time-1)
            self.linear = nn.Linear(1, 1)

        self.orig_d_ts = orig_d_ts
        self.d_ts = embed_dim
        self.ts_seq_num = ts_seq_num

        if self.irregular_learn_emb_ts:
            self.time_attn_ts = multiTimeAttention(
                self.orig_d_ts*2, self.d_ts, embed_time, 8)

        if self.reg_ts:
            self.orig_reg_d_ts = orig_reg_d_ts
            self.proj_ts = nn.Conv1d(self.orig_reg_d_ts, self.d_ts, kernel_size=self.kernel_size, padding=math.floor(
                (self.kernel_size - 1) / 2), bias=False)

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

        Biobert = AutoModel.from_pretrained(model_name)
        self.orig_d_txt = orig_d_txt
        self.d_txt = embed_dim
        self.text_seq_num = text_seq_num
        self.bertrep = BertForRepresentation(model_name, Biobert)

        if self.irregular_learn_emb_text:
            self.time_attn = multiTimeAttention(
                768, self.d_txt, embed_time, 8)
        else:
            self.proj_txt = nn.Conv1d(self.orig_d_txt, self.d_txt, kernel_size=self.kernel_size, padding=math.floor(
                (self.kernel_size - 1) / 2), bias=False)

        output_dim = num_labels
        # if self.modeltype == "TS_Text":
        #     if self.cross_method == "self_cross":
        #         self.trans_self_cross_ts_txt = self.get_cross_network(
        #             layers=cross_layers)
        #         self.proj1 = nn.Linear(
        #             self.d_ts+self.d_txt, self.d_ts+self.d_txt)
        #         self.proj2 = nn.Linear(
        #             self.d_ts+self.d_txt, self.d_ts+self.d_txt)
        #         self.out_layer = nn.Linear(self.d_ts+self.d_txt, output_dim)
        #     else:
        #         self.trans_ts_mem = self.get_network(
        #             self_type='ts_mem', layers=layers)
        #         self.trans_txt_mem = self.get_network(
        #             self_type='txt_mem', layers=layers)

        #         if self.cross_method == "MulT":
        #             self.trans_txt_with_ts = self.get_network(
        #                 self_type='txt_with_ts', layers=cross_layers)
        #             self.trans_ts_with_txt = self.get_network(
        #                 self_type='ts_with_txt', layers=cross_layers)
        #             self.proj1 = nn.Linear(
        #                 (self.d_ts+self.d_txt), (self.d_ts+self.d_txt))
        #             self.proj2 = nn.Linear(
        #                 (self.d_ts+self.d_txt), (self.d_ts+self.d_txt))
        #             self.out_layer = nn.Linear(
        #                 (self.d_ts+self.d_txt), output_dim)
        #         elif self.cross_method == "MAGGate":
        #             self.gate_fusion = MAGGate(
        #                 inp1_size=self.d_txt, inp2_size=self.d_ts, dropout=self.embed_dropout)
        #             self.proj1 = nn.Linear(self.d_txt, self.d_txt)
        #             self.proj2 = nn.Linear(self.d_txt, self.d_txt)
        #             self.out_layer = nn.Linear(self.d_txt, output_dim)
        #         elif self.cross_method == "Outer":
        #             self.outer_fusion = Outer(
        #                 inp1_size=self.d_txt, inp2_size=self.d_ts)
        #             self.proj1 = nn.Linear(self.d_txt, self.d_txt)
        #             self.proj2 = nn.Linear(self.d_txt, self.d_txt)
        #             self.out_layer = nn.Linear(self.d_txt, output_dim)
        #         else:
        #             self.proj1 = nn.Linear(
        #                 self.d_ts+self.d_txt, self.d_ts+self.d_txt)
        #             self.proj2 = nn.Linear(
        #                 self.d_ts+self.d_txt, self.d_ts+self.d_txt)
        #             self.out_layer = nn.Linear(
        #                 self.d_ts+self.d_txt, output_dim)
        # elif self.modeltype == "TS":
        #     self.proj1 = nn.Linear(self.d_ts, self.d_ts)
        #     self.proj2 = nn.Linear(self.d_ts, self.d_ts)
        #     self.out_layer = nn.Linear(self.d_ts, output_dim)
        # elif self.modeltype == "Text":
        #     self.proj1 = nn.Linear(self.d_txt, self.d_txt)
        #     self.proj2 = nn.Linear(self.d_txt, self.d_txt)
        #     self.out_layer = nn.Linear(self.d_txt, output_dim)
        # else:
        #     raise NotImplementedError

        # define multi-scale convolution
        self.ts_conv1 = nn.Conv1d(self.d_ts,
                                  self.d_ts, kernel_size=4, stride=2)
        self.ts_conv2 = nn.Conv1d(self.d_ts,
                                  self.d_ts, kernel_size=8, stride=4)
        self.ts_conv3 = nn.Conv1d(self.d_ts,
                                  self.d_ts, kernel_size=16, stride=8)
        self.text_conv1 = nn.Conv1d(self.d_txt,
                                    self.d_txt, kernel_size=4, stride=2)
        self.text_conv2 = nn.Conv1d(self.d_txt,
                                    self.d_txt, kernel_size=8, stride=4)
        self.text_conv3 = nn.Conv1d(self.d_txt,
                                    self.d_txt, kernel_size=16, stride=8)

        self.proj1 = nn.Linear(self.d_ts+self.d_txt, self.d_ts+self.d_txt)
        self.proj2 = nn.Linear(self.d_ts+self.d_txt, self.d_ts+self.d_txt)
        self.out_layer = nn.Linear(self.d_ts+self.d_txt, output_dim)

        if self.task == 'ihm':
            self.loss_fct1 = nn.CrossEntropyLoss()
        elif self.task == 'pheno':
            self.loss_fct1 = nn.BCEWithLogitsLoss()
        else:
            raise ValueError("Unknown task")

        # assume 50 concepts are highly related to the task
        self.scale1_concepts = nn.Parameter(torch.zeros(
            50, embed_dim), requires_grad=True)
        nn.init.trunc_normal_(self.scale1_concepts, std=1.0 /
                              math.sqrt(embed_dim))

        self.scale2_concepts = nn.Parameter(torch.zeros(
            30, embed_dim), requires_grad=True)
        nn.init.trunc_normal_(self.scale2_concepts, std=1.0 /
                              math.sqrt(embed_dim))

        self.scale3_concepts = nn.Parameter(torch.zeros(
            20, embed_dim), requires_grad=True)
        nn.init.trunc_normal_(self.scale3_concepts, std=1.0 /
                              math.sqrt(embed_dim))

        self.ot_coattn = OT_Attn_assem(
            impl="pot-uot-l2", ot_reg=0.05, ot_tau=0.5)

    def learn_time_embedding(self, tt):
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)

    def forward(self, x_ts, x_ts_mask, ts_tt_list,
                input_ids_sequences, attn_mask_sequences,
                note_time_list, note_time_mask_list,
                labels=None, reg_ts=None):
        """ Forward function of Multimodal model
        :param x_ts: (B, N, D_t), torch.Tensor, time series data
        :param x_ts_mask: (B, N, D_t), torch.Tensor, time series mask
        :param ts_tt_list: (B, N), torch.Tensor, time series time
        :param input_ids_sequences: (B, N_text, L_text), torch.Tensor, text data
        :param attn_mask_sequences: (B, N_text, L_text), torch.Tensor, text mask
        :param note_time_list: (B, N_text), torch.Tensor, time of text data
        :param note_time_mask_list: (B, N_text), torch.Tensor, mask of text data
        :param labels: (B, ), torch.Tensor, labels
        :param reg_ts: (B, N_r, D_r), torch.Tensor, regular time series data
        """
        if self.irregular_learn_emb_ts:
            # (B, N) -> (B, N, embed_time)
            time_key_ts = self.learn_time_embedding(
                ts_tt_list)
            # (1, N_r) -> (1, N_r, embed_time)
            time_query = self.learn_time_embedding(
                self.time_query.unsqueeze(0).type_as(x_ts))

            x_ts_irg = torch.cat((x_ts, x_ts_mask), 2)
            x_ts_mask = torch.cat((x_ts_mask, x_ts_mask), 2)

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

        if self.reg_ts and reg_ts != None:
            # convolution over regular time series
            x_ts_reg = reg_ts.transpose(1, 2)
            proj_x_ts_reg = x_ts_reg if self.orig_reg_d_ts == self.d_ts else self.proj_ts(
                x_ts_reg)
            proj_x_ts_reg = proj_x_ts_reg.permute(2, 0, 1)
        else:
            raise ValueError("Not implemented")

        if self.TS_mixup:
            if self.mixup_level == 'batch':
                g_irg = torch.max(proj_x_ts_irg, dim=0).values
                g_reg = torch.max(proj_x_ts_reg, dim=0).values
                moe_gate = torch.cat([g_irg, g_reg], dim=-1)
            elif self.mixup_level == 'batch_seq' or self.mixup_level == 'batch_seq_feature':
                moe_gate = torch.cat(
                    [proj_x_ts_irg, proj_x_ts_reg], dim=-1)
            else:
                raise ValueError("Unknown mixedup type")
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

        # -> (B, N_text, D_text)
        x_txt = self.bertrep(input_ids_sequences, attn_mask_sequences)
        if self.irregular_learn_emb_text:
            # (B, N_text) -> (B, N_text, embed_time)
            time_key = self.learn_time_embedding(
                note_time_list)
            if not self.irregular_learn_emb_ts:
                time_query = self.learn_time_embedding(
                    self.time_query.unsqueeze(0))
            # (B, N_r, embed_time)
            proj_x_txt = self.time_attn(
                time_query, time_key, x_txt, note_time_mask_list)
            proj_x_txt = proj_x_txt.transpose(0, 1)
        else:
            x_txt = x_txt.transpose(1, 2)
            proj_x_txt = x_txt if self.orig_d_txt == self.d_txt else self.proj_txt(
                x_txt)
            proj_x_txt = proj_x_txt.permute(2, 0, 1)

        # proj_x_ts: (TT_max, B, D_t)
        # proj_x_txt: (TT_max, B, D_text)
        # expected output: (B, D_t + D_text)

        # Method 1: mean pooling and concat
        mean_ts = torch.mean(proj_x_ts, dim=0)
        mean_text = torch.mean(proj_x_txt, dim=0)

        # if torch.isnan(output).any():
        #     ipdb.set_trace()

        # # extract multi-scale ts features
        # ts_feat1 = self.ts_conv1(
        #     rearrange(proj_x_ts, "tt b d -> b d tt"))  # B, 128, 12
        # ts_feat2 = self.ts_conv2(
        #     rearrange(proj_x_ts, "tt b d -> b d tt"))  # B, 128, 6
        # ts_feat3 = self.ts_conv3(
        #     rearrange(proj_x_ts, "tt b d -> b d tt"))  # B, 128, 3

        # # extract multi-scale text features
        # text_feat1 = self.text_conv1(
        #     rearrange(proj_x_txt, "tt b d -> b d tt"))  # B, 128, 12
        # text_feat2 = self.text_conv2(
        #     rearrange(proj_x_txt, "tt b d -> b d tt"))  # B, 128, 6
        # text_feat3 = self.text_conv3(
        #     rearrange(proj_x_txt, "tt b d -> b d tt"))  # B, 128, 3

        # rec_ts_emb1 = self.recon_prototype_embeds(ts_feat1)
        # rec_text_emb1 = self.recon_prototype_embeds(text_feat1)
        # rec_ts_emb2 = self.recon_prototype_embeds(ts_feat2)
        # rec_text_emb2 = self.recon_prototype_embeds(text_feat2)
        # rec_ts_emb3 = self.recon_prototype_embeds(ts_feat3)
        # rec_text_emb3 = self.recon_prototype_embeds(text_feat3)

        # if self.task == 'ihm':
        #     if labels != None:
        #         return self.loss_fct1(output, labels)
        #     return F.softmax(output, dim=-1)[:, 1]

        # elif self.task == 'pheno':
        #     if labels != None:
        #         labels = labels.float()
        #         return self.loss_fct1(output, labels)
        #     return torch.sigmoid(output)

    def recon_prototype_embeds(self, ts_feat1: torch.Tensor) -> torch.Tensor:
        '''
        ts_feat_1: (B, D, TT)
        prototype_embeds: (B, D)
        '''
        batch_size = ts_feat1.size(0)
        flatten_ts_feat1 = rearrange(ts_feat1, "b d tt -> (b tt) d")
        ts_atten_scale1, _ = self.ot_coattn(
            flatten_ts_feat1, self.scale1_concepts)
        ts_prob_scale1 = F.softmax(ts_atten_scale1, dim=-1)
        rec_ts_feat1 = rearrange(
            ts_prob_scale1 @ self.scale1_concepts, "(b tt) d -> b d tt", b=batch_size)
        # prototype-oriented pooling
        rec_ts_emb1 = rec_ts_feat1.mean(dim=-1)

        return rec_ts_emb1

    def pred_logits(self, ts_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        last_hs = torch.cat([ts_emb, text_emb], dim=-1)
        last_hs_proj = self.proj2(
            F.dropout(F.relu(self.proj1(last_hs)), p=self.dropout, training=self.training))
        last_hs_proj += last_hs
        output = self.out_layer(last_hs_proj)
        return output


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from cmehr.paths import *
    from cmehr.dataset.UTDE_datamodule import TSNote_Irg, TextTSIrgcollate_fn

    dataset = TSNote_Irg(
        file_path=str(ROOT_PATH / "output/ihm"),
        split="train",
        bert_type="yikuan8/Clinical-Longformer",
        max_length=128
    )

    dataloader = DataLoader(dataset=dataset,
                            batch_size=4,
                            num_workers=1,
                            shuffle=True,
                            collate_fn=TextTSIrgcollate_fn)
    assert len(dataloader) > 0
    batch = dict()
    for batch in dataloader:
        break

    for k, v in batch.items():  # type: ignore
        print(k, ": ", v.shape)

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
    model = ProtoMULT()
    loss = model(
        x_ts=batch["ts"],  # type ignore
        x_ts_mask=batch["ts_mask"],
        ts_tt_list=batch["ts_tt"],
        reg_ts=batch["reg_ts"],
        input_ids_sequences=batch["input_ids"],
        attn_mask_sequences=batch["attention_mask"],
        note_time_list=batch["note_time"],
        note_time_mask_list=batch["note_time_mask"],
        labels=batch["label"],
    )
    print(loss)

    # feat1 = torch.randn(12, 128)
    # feat2 = torch.randn(48, 128)
    # coattn = OT_Attn_assem(impl="pot-uot-l2", ot_reg=0.05, ot_tau=0.5)
    # A_coattn, dist = coattn(feat1, feat2)
    # A_coattn: optimal transport matrix feat1 -> feat2
    ipdb.set_trace()
