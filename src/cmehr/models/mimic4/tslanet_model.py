import torch
import torch.nn as nn
import torch.nn.functional as F
# import pytorch_pretrained_bert as Bert
from timm.models.layers import DropPath
from timm.models.layers import trunc_normal_
from cmehr.models.mimic4.base_model import MIMIC4LightningModule

import ipdb


class ICB(nn.Module):
    # interactive convolution block
    def __init__(self, in_features, hidden_features, drop=0.):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, hidden_features, 1)
        self.conv2 = nn.Conv1d(in_features, hidden_features, 3, 1, 1)
        self.conv3 = nn.Conv1d(hidden_features, in_features, 1)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()

    def forward(self, x):
        x = x.transpose(1, 2)
        x1 = self.conv1(x)
        x1_1 = self.act(x1)
        x1_2 = self.drop(x1_1)

        x2 = self.conv2(x)
        x2_1 = self.act(x2)
        x2_2 = self.drop(x2_1)

        out1 = x1 * x2_2
        out2 = x2 * x1_2

        x = self.conv3(out1 + out2)
        x = x.transpose(1, 2)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, seq_len, patch_size=4, in_chans=30, embed_dim=128):
        super().__init__()
        stride = max(1, patch_size // 2)
        num_patches = int((seq_len - patch_size) / stride + 1)
        self.num_patches = num_patches
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)

    def forward(self, x):
        x_out = self.proj(x).flatten(2).transpose(1, 2)
        return x_out


class Adaptive_Spectral_Block(nn.Module):
    # Adaptive Spectral Block
    def __init__(self, dim, adaptive_filter=True):
        super().__init__()
        self.complex_weight_high = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        self.complex_weight = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)

        trunc_normal_(self.complex_weight_high, std=.02)
        trunc_normal_(self.complex_weight, std=.02)
        self.threshold_param = nn.Parameter(torch.rand(1) * 0.5)
        self.adaptive_filter = adaptive_filter

    def create_adaptive_high_freq_mask(self, x_fft):
        B, _, _ = x_fft.shape

        # Calculate energy in the frequency domain
        energy = torch.abs(x_fft).pow(2).sum(dim=-1)

        # Flatten energy across H and W dimensions and then compute median
        flat_energy = energy.view(B, -1)  # Flattening H and W into a single dimension
        median_energy = flat_energy.median(dim=1, keepdim=True)[0]  # Compute median
        median_energy = median_energy.view(B, 1)  # Reshape to match the original dimensions

        # Normalize energy
        epsilon = 1e-6  # Small constant to avoid division by zero
        normalized_energy = energy / (median_energy + epsilon)

        threshold = torch.quantile(normalized_energy, self.threshold_param)
        dominant_frequencies = normalized_energy > threshold

        # Initialize adaptive mask
        adaptive_mask = torch.zeros_like(x_fft, device=x_fft.device)
        adaptive_mask[dominant_frequencies] = 1

        return adaptive_mask

    def forward(self, x_in):
        B, N, C = x_in.shape

        dtype = x_in.dtype
        x = x_in.to(torch.float32)

        # Apply FFT along the time dimension
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x_weighted = x_fft * weight

        if self.adaptive_filter:
            # Adaptive High Frequency Mask (no need for dimensional adjustments)
            freq_mask = self.create_adaptive_high_freq_mask(x_fft)
            x_masked = x_fft * freq_mask.to(x.device)

            weight_high = torch.view_as_complex(self.complex_weight_high)
            x_weighted2 = x_masked * weight_high

            x_weighted += x_weighted2

        # Apply Inverse FFT
        x = torch.fft.irfft(x_weighted, n=N, dim=1, norm='ortho')

        x = x.to(dtype)
        x = x.view(B, N, C)  # Reshape back to original shape

        return x


class TSLANet_layer(nn.Module):
    def __init__(self, dim, mlp_ratio=3., drop=0., drop_path=0., norm_layer=nn.LayerNorm,
                 use_ASB=True, use_ICB=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.asb = Adaptive_Spectral_Block(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.icb = ICB(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        self.ASB = use_ASB
        self.ICB = use_ICB

    def forward(self, x):
        # Check if both ASB and ICB are true
        if self.ICB and self.ASB:
            x = x + self.drop_path(self.icb(self.norm2(self.asb(self.norm1(x)))))
        # If only ICB is true
        elif self.ICB:
            x = x + self.drop_path(self.icb(self.norm2(x)))
        # If only ASB is true
        elif self.ASB:
            x = x + self.drop_path(self.asb(self.norm1(x)))
        # If neither is true, just pass x through
        return x
    

class TSLANETModule(MIMIC4LightningModule):
    def __init__(self,
                 task: str = "ihm",
                 modeltype: str = "TS",
                 max_epochs: int = 10,
                 img_learning_rate: float = 1e-4,
                 ts_learning_rate: float = 4e-4,
                 period_length: int = 48,
                 orig_reg_d_ts: int = 30,
                 hidden_dim=128,
                 *args,
                 **kwargs):
        super().__init__(task=task, modeltype=modeltype, max_epochs=max_epochs,
                         img_learning_rate=img_learning_rate, ts_learning_rate=ts_learning_rate,
                         period_length=period_length)

        self.input_size = orig_reg_d_ts
        self.hidden_dim = hidden_dim

        self.patch_size = 3
        self.dropout_rate = 0.15
        self.depth = 2
        self.patch_embed = PatchEmbed(
            seq_len=self.tt_max, patch_size=self.patch_size,
            in_chans=self.input_size, embed_dim=self.hidden_dim
        )
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.hidden_dim), requires_grad=True)
        self.pos_drop = nn.Dropout(p=self.dropout_rate)

        self.input_layer = nn.Linear(self.patch_size, self.hidden_dim)

        dpr = [x.item() for x in torch.linspace(0, self.dropout_rate, self.depth)]  # stochastic depth decay rule

        self.tsla_blocks = nn.ModuleList([
            TSLANet_layer(dim=self.hidden_dim, drop=self.dropout_rate, drop_path=dpr[i])
            for i in range(self.depth)]
        )

        # Classifier head
        self.head = nn.Linear(self.hidden_dim, self.num_labels)

        # init weights
        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self,
                reg_ts,
                labels=None,
                **kwargs):
        
        batch_size = reg_ts.size(0)
        x = reg_ts.transpose(1, 2)

        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for tsla_blk in self.tsla_blocks:
            x = tsla_blk(x)

        x = x.mean(1)
        output = self.head(x)

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
    model = TSLANETModule(
    )
    loss = model(
        reg_ts=batch["reg_ts"],
        labels=batch["label"]
    )
    print(loss)
