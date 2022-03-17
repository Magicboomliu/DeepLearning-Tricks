import torch
from torch import nn, einsum
import numpy as np
from einops import rearrange, repeat


class CyclicShift(nn.Module):
    def __init__(self, displacement):
        super().__init__()
        self.displacement = displacement

    def forward(self, x):
        return torch.roll(x, shifts=(self.displacement, self.displacement), dims=(1, 2))

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, drop_rate=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=drop_rate),
        )

    def forward(self, x):
        return self.net(x)

class WindowAttention(nn.Module):
    def __init__(self, dim, head_dim, shifted, window_size, drop_rate=0.1):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.head_num = self.dim // self.head_dim
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.shifted = shifted
        self.drop = nn.Dropout(p=drop_rate)
        self.register_buffer("relative_indices", self.get_relative_distances(window_size))
        self.to_qkv = nn.Linear(self.dim, self.dim * 3, bias=False)
        self.pos_embedding = nn.Parameter(torch.randn(2*window_size-1, 2*window_size-1, self.head_num) )
        self.to_out = nn.Linear(self.dim, self.dim)
        if self.shifted:
            displacement = window_size // 2
            self.cyclic_shift = CyclicShift(-displacement)
            self.cyclic_back_shift = CyclicShift(displacement)
            self.register_buffer(
                "upper_lower_mask",
                self.create_mask(
                    window_size=window_size, displacement=displacement,
                    upper_lower=True, left_right=False
                )
            )
            self.register_buffer(
                "left_right_mask",
                self.create_mask(
                    window_size=window_size, displacement=displacement,
                    upper_lower=False, left_right=True
                )
            )


    def get_relative_distances(self, window_size):
        indices = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
        distances = indices[None, :, :] - indices[:, None, :]
        return distances

    def create_mask(self, window_size, displacement, upper_lower, left_right):
        mask = torch.zeros(window_size ** 2, window_size ** 2)
        mask = rearrange(mask, '(h1 w1) (h2 w2) -> h1 w1 h2 w2', h1=window_size, h2=window_size)
        if upper_lower:
            mask[:-displacement, :, -displacement:, :] = float("-inf")
            mask[-displacement:, :, :-displacement, :] = float("-inf")
        if left_right:
            mask[:, -displacement:, :, :-displacement] = float('-inf')
            mask[:, :-displacement, :, -displacement:] = float('-inf')
        mask = rearrange(mask, 'h1 w1 h2 w2 -> (h1 w1) (h2 w2)')
        return mask

    def forward(self, x):
        if self.shifted:
            x = self.cyclic_shift(x)
        _, feature_map_h, feature_map_w, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        nw_h = feature_map_h // self.window_size
        nw_w = feature_map_w // self.window_size
        q, k, v = map(
            lambda t: rearrange(
                t, 'b (nw_h w_h) (nw_w w_w) (h d) -> b h (nw_h nw_w) (w_h w_w) d',
                h=self.head_num, w_h=self.window_size, w_w=self.window_size
            ),
            qkv,
        )
        dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale
        dots += rearrange(self.pos_embedding[self.relative_indices[:, :, 0], self.relative_indices[:, :, 1], :], "a b h -> 1 h 1 a b")
        if self.shifted:
            dots[:, :, -nw_w:] += self.upper_lower_mask
            dots[:, :, nw_w - 1::nw_w] += self.left_right_mask
        attn = dots.softmax(dim=-1)
        attn = self.drop(attn)
        out = einsum('b h w i j, b h w j d -> b h w i d', attn, v)
        out = rearrange(out, 'b h (nw_h nw_w) (w_h w_w) d -> b (nw_h w_h) (nw_w w_w) (h d)',
                        h=self.head_num, w_h=self.window_size, w_w=self.window_size, nw_h=nw_h, nw_w=nw_w)
        out = self.to_out(out)
        out = self.drop(out)
        if self.shifted:
            out = self.cyclic_back_shift(out)
        return out

class SwinBlock(nn.Module):
    def __init__(self, dim, head_dim, mlp_dim, shifted, window_size, drop_rate):
        super().__init__()
        self.attention_block = Residual(PreNorm(dim, WindowAttention(dim=dim, head_dim=head_dim, shifted=shifted, window_size=window_size, drop_rate=drop_rate)))
        self.mlp_block = Residual(PreNorm(dim, FeedForward(dim=dim, hidden_dim=mlp_dim, drop_rate=drop_rate)))

    def forward(self, x):
        x = self.attention_block(x)
        x = self.mlp_block(x)
        return x

class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, downscaling_factor):
        super().__init__()
        self.downscaling_factor = downscaling_factor
        self.patch_merge = nn.Unfold(kernel_size=downscaling_factor, stride=downscaling_factor, padding=0)
        self.linear = nn.Linear(in_channels * downscaling_factor ** 2, out_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        new_h, new_w = h // self.downscaling_factor, w // self.downscaling_factor
        x = self.patch_merge(x).view(b, -1, new_h, new_w).permute(0, 2, 3, 1)
        x = self.linear(x)
        return x

class StageModule(nn.Module):
    def __init__(self, in_channels, dim, layer_num, downscaling_factor, head_dim, window_size, drop_rate):
        super().__init__()
        assert layer_num % 2 == 0, 'Stage layer_num need to be divisible by 2 for regular and shifted block.'

        self.patch_partition = PatchMerging(in_channels=in_channels, out_channels=dim, downscaling_factor=downscaling_factor)
        self.layer_num = nn.ModuleList([])
        for _ in range(layer_num // 2):
            self.layer_num.append(nn.ModuleList([
                SwinBlock(dim=dim, head_dim=head_dim, mlp_dim=dim * 4, shifted=False, window_size=window_size, drop_rate=drop_rate),
                SwinBlock(dim=dim, head_dim=head_dim, mlp_dim=dim * 4, shifted=True, window_size=window_size, drop_rate=drop_rate),
            ]))

    def forward(self, x):
        x = self.patch_partition(x)
        for regular_block, shifted_block in self.layer_num:
            x = regular_block(x)
            x = shifted_block(x)
        return x.permute(0, 3, 1, 2)

class SwinTransformer(nn.Module):
    def __init__(self, args, pretrain=False):
        super().__init__()
        self.args = args
        self.swin_dim = self.args.swin_dim
        self.swin_head_dim = self.args.swin_head_dim
        self.swin_window_size = self.args.swin_window_size
        self.swin_layer_num_list = self.args.swin_layer_num_list
        self.swin_downscaling_factors = self.args.swin_downscaling_factors
        self.drop_rate = self.args.drop_rate
        self.stage1 = StageModule(
            in_channels=3,
            dim=self.swin_dim,
            layer_num=self.swin_layer_num_list[0],
            downscaling_factor=self.swin_downscaling_factors[0],
            head_dim=self.swin_head_dim,
            window_size=self.swin_window_size,
            drop_rate=self.drop_rate,
        )
        self.stage2 = StageModule(
            in_channels=self.swin_dim,
            dim=self.swin_dim * 2,
            layer_num=self.swin_layer_num_list[1],
            downscaling_factor=self.swin_downscaling_factors[1],
            head_dim=self.swin_head_dim,
            window_size=self.swin_window_size,
            drop_rate=self.drop_rate,
        )
        self.stage3 = StageModule(
            in_channels=self.swin_dim * 2,
            dim=self.swin_dim * 4,
            layer_num=self.swin_layer_num_list[2],
            downscaling_factor=self.swin_downscaling_factors[2],
            head_dim=self.swin_head_dim,
            window_size=self.swin_window_size,
            drop_rate=self.drop_rate,
        )
        self.stage4 = StageModule(
            in_channels=self.swin_dim * 4,
            dim=self.swin_dim * 8,
            layer_num=self.swin_layer_num_list[3],
            downscaling_factor=self.swin_downscaling_factors[3],
            head_dim=self.swin_head_dim,
            window_size=self.swin_window_size,
            drop_rate=self.drop_rate,
        )
        if pretrain:
            self.load_pretrain_param()

    def load_pretrain_param(self):
        pass


    def forward(self, img):           # [batch_size, 3, h, w]                          [b, 3, 416, 416]
        out1 = self.stage1(img)       # [batch_size, hidden_dim, h//4, w//4]           [b, 96, 104, 104]
        out2 = self.stage2(out1)      # [batch_size, hidden_dim*2, h//8, w//8]         [b, 192, 52, 52]
        out3 = self.stage3(out2)      # [batch_size, hidden_dim*4, h//16, w//16]       [b, 384, 26, 26]
        out4 = self.stage4(out3)      # [batch_size, hidden_dim*8, h//32, w//32]       [b, 768, 13, 13]
        out_list = [out4, out3, out2, out1]
        return out_list
