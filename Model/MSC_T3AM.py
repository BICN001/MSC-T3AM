import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from Attention import FT_A


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.wise1 = nn.Sequential(
            nn.Conv2d(num_heads, num_heads, (25, 1), groups=num_heads, padding='same'),
            nn.BatchNorm2d(num_heads),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(num_heads, emb_size, (1, 1)),
            nn.BatchNorm2d(emb_size),
            nn.ReLU(),
            nn.AvgPool2d((2, 1)),
            nn.Dropout(0.3),
        )
        self.wise2 = nn.Sequential(
            nn.Conv2d(num_heads, num_heads, (75, 1), groups=num_heads, padding='same'),
            nn.BatchNorm2d(num_heads),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(num_heads, emb_size, (1, 1)),
            nn.BatchNorm2d(emb_size),
            nn.ReLU(),
            nn.AvgPool2d((2, 1)),
            nn.Dropout(0.3),
        )
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size * (emb_size // num_heads), emb_size)

    def MSC(self, x):
        x1 = self.wise1(x)
        x2 = self.wise2(x)
        x = torch.cat([x1, x2], 2)
        return x

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        queries = self.MSC(queries)
        keys = self.MSC(keys)
        values = self.MSC(values)

        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size, num_heads=10, drop_p=0.5, forward_expansion=4, forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])


class MSC_T3AM(nn.Module):
    def __init__(self, emb_size=10, depth=3, n_classes=6, channel=62, timepoint=1500, **kwargs):
        super(MSC_T3AM, self).__init__()

        self.ch = channel
        self.time = timepoint
        self.channel_convolution = nn.Sequential(
            nn.Conv2d(emb_size, emb_size, (channel, 1), groups=emb_size),
            nn.BatchNorm2d(emb_size),
            nn.ELU(),
            nn.AvgPool2d((1, 10)),
            nn.Dropout(0.5),
        )
        self.FT_A = FT_A(emb_size)
        self.projection = nn.Sequential(
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.MSC_Transformer = TransformerEncoder(depth, emb_size)
        self.fc = nn.Sequential(
            nn.Linear(1500, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes)
        )
        self.channel_weight = nn.Parameter(torch.randn(emb_size, 1, channel), requires_grad=True)
        nn.init.xavier_uniform_(self.channel_weight.data)

    def forward(self, x):
        x = x.view(-1, 1, self.ch, self.time)
        x = torch.einsum('bdcw, hdc->bhcw', x, self.channel_weight)
        x = self.channel_convolution(x)
        x = self.FT_A(x)
        x = self.projection(x)
        x = self.MSC_Transformer(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out
