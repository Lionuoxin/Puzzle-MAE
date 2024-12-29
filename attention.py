import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from positon import get_relative_position_cpb, get_relative_position_cpb_2D
from cope import CoPE, CoPE2D
import unfoldNd

@torch.no_grad()
def get_seqlen_and_mask(input_resolution, window_size):
    unfold_op = unfoldNd.UnfoldNd(
        (3, 3, 3), dilation=1, padding=(window_size // 2, window_size // 2, window_size // 2), stride=(1, 1, 1)
    )
    # attn_map = F.unfold(torch.ones([1, 1, input_resolution[0], input_resolution[1]]), window_size,
    #                     dilation=1, padding=(window_size // 2, window_size // 2, ), stride=1)
    attn_map = unfold_op(torch.ones([1, 1, input_resolution[0], input_resolution[1], input_resolution[2]]))
    attn_local_length = attn_map.sum(-2).squeeze().unsqueeze(-1)
    attn_mask = (attn_map.squeeze(0).permute(1, 0)) == 0
    return attn_local_length, attn_mask


class AggregatedAttention(nn.Module):
    def __init__(self, dim, input_resolution, num_heads=8, window_size=3, qkv_bias=True,
                 attn_drop=0., proj_drop=0., sr_ratio=1, t_ratio=1, fixed_pool_size=None):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.input_resolution = input_resolution

        self.sr_ratio = sr_ratio
        self.t_ratio = t_ratio

        assert window_size % 2 == 1, "window size must be odd"
        self.window_size = window_size
        self.local_len = window_size ** 3

        if fixed_pool_size is None:
            self.pool_H, self.pool_W, self.pool_T = input_resolution[0] // self.sr_ratio, input_resolution[1] // self.sr_ratio, input_resolution[2] // self.t_ratio
        else:
            assert fixed_pool_size < min(input_resolution), \
                f"The fixed_pool_size {fixed_pool_size} should be less than the shorter side of input resolution {input_resolution} to ensure pooling works correctly."
            self.pool_H, self.pool_W, self.pool_T = fixed_pool_size, fixed_pool_size, fixed_pool_size
        self.pool_len = self.pool_H * self.pool_W * self.pool_T

        # self.unfold = nn.Unfold(kernel_size=window_size, padding=window_size // 2, stride=1)
        self.unfold = unfoldNd.UnfoldNd(
            kernel_size=(3, 3, 3), dilation=1, padding=(window_size // 2, window_size // 2, window_size // 2), stride=(1, 1, 1)
        )

        self.softmax = nn.Softmax(dim=-1)
        self.scale = self.head_dim ** -0.5

        self.temperature = nn.Parameter(
            torch.log((torch.ones(num_heads, 1, 1) / 0.24).exp() - 1))  # Initialize softplus(temperature) to 1/0.24.

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.query_embedding = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(self.num_heads, 1, self.head_dim), mean=0, std=0.02))
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Components to generate pooled features.
        self.pool = nn.AdaptiveAvgPool3d((self.pool_H, self.pool_W, self.pool_T))
        self.sr = nn.Conv3d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()

        self.cope = CoPE(input_resolution[0]*input_resolution[1]*input_resolution[2], self.head_dim, self.pool_H, self.pool_W, self.pool_T)

        # mlp to generate continuous relative position bias
        self.cpb_fc1 = nn.Linear(3, 512, bias=True)
        self.cpb_act = nn.ReLU(inplace=True)
        self.cpb_fc2 = nn.Linear(512, num_heads, bias=True)

        # relative bias for local features
        self.relative_pos_bias_local = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(num_heads, self.local_len), mean=0, std=0.0004))

        # Generate padding_mask && sequnce length scale
        local_seq_length, padding_mask = get_seqlen_and_mask(input_resolution, window_size)
        self.register_buffer("seq_length_scale", torch.as_tensor(np.log(local_seq_length.numpy() + self.pool_len)),
                             persistent=False)
        self.register_buffer("padding_mask", padding_mask, persistent=False)


    def forward(self, x, H, W, T, relative_pos_index, relative_coords_table):
        B, N, C = x.shape

        q_norm = F.normalize(self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3), dim=-1)
        q_norm = (q_norm + self.query_embedding) * F.softplus(self.temperature) * self.seq_length_scale

        k_local, v_local = self.kv(x).chunk(2, dim=-1)
        k_local = F.normalize(k_local.reshape(B, N, self.num_heads, self.head_dim), dim=-1).reshape(B, N, -1)
        
        k_local, v_local = self.unfold(torch.cat([k_local, v_local], dim=-1).permute(0, 2, 1).reshape(B, -1, H, W, T)).reshape(
            B, 2 * self.num_heads, self.head_dim, self.local_len, N).permute(0, 1, 4, 2, 3).chunk(2, dim=1)
        
        x_local = (torch.softmax(((q_norm.unsqueeze(-2) @ k_local).squeeze(-2) \
                      + self.relative_pos_bias_local.unsqueeze(1)).masked_fill(self.padding_mask, float('-inf')), dim=-1).unsqueeze(-2) 
                   @ v_local.transpose(-2, -1)).squeeze(-2)
        
        x = x.permute(0, 2, 1).reshape(B, -1, H, W, T).contiguous()
        x = self.pool(self.act(self.sr(x))).reshape(B, -1, self.pool_len).permute(0, 2, 1)
        x = self.norm(x)

        
        k_pool, v_pool = self.kv(x).reshape(B, self.pool_len, 2 * self.num_heads, self.head_dim).permute(0, 2, 1, 3).chunk(2, dim=1)

        pool_bias = self.cpb_fc2(self.cpb_act(self.cpb_fc1(relative_coords_table))).transpose(0, 1)[:,
                    relative_pos_index.view(-1)].view(-1, N, self.pool_len)
        
        x_pool = (torch.softmax((q_norm @ F.normalize(k_pool, dim=-1).transpose(-2, -1) + pool_bias + self.cope(q_norm, q_norm @ F.normalize(k_pool, dim=-1).transpose(-2, -1) + pool_bias)), dim=-1)) @ v_pool

        x = (x_local + x_pool).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x
    
def get_seqlen_and_mask_2d(input_resolution, window_size):
    unfold_op = unfoldNd.UnfoldNd(
        (3, 3), dilation=1, padding=(window_size // 2, window_size // 2), stride=(1, 1)
    )
    attn_map = unfold_op(torch.ones([1, 1, input_resolution[0], input_resolution[1]]))
    attn_local_length = attn_map.sum(-2).squeeze().unsqueeze(-1)
    attn_mask = (attn_map.squeeze(0).permute(1, 0)) == 0
    return attn_local_length, attn_mask

class AggregatedAttention2D(nn.Module):
    def __init__(self, dim, input_resolution, num_heads=8, window_size=3, qkv_bias=True,
                 attn_drop=0., proj_drop=0., sr_ratio=1, fixed_pool_size=None):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.sr_ratio = sr_ratio

        assert window_size % 2 == 1, "window size must be odd"
        self.window_size = window_size
        self.local_len = window_size ** 2

        if fixed_pool_size is None:
            self.pool_H, self.pool_W = input_resolution[0] // self.sr_ratio, input_resolution[1] // self.sr_ratio
        else:
            assert fixed_pool_size < min(input_resolution), \
                f"The fixed_pool_size {fixed_pool_size} should be less than the shorter side of input resolution {input_resolution} to ensure pooling works correctly."
            self.pool_H, self.pool_W = fixed_pool_size, fixed_pool_size, fixed_pool_size
        self.pool_len = self.pool_H * self.pool_W

        # self.unfold = nn.Unfold(kernel_size=window_size, padding=window_size // 2, stride=1)
        self.unfold = unfoldNd.UnfoldNd(
            kernel_size=(3, 3), dilation=1, padding=(window_size // 2, window_size // 2), stride=(1, 1)
        )

        self.softmax = nn.Softmax(dim=-1)
        self.scale = self.head_dim ** -0.5

        self.temperature = nn.Parameter(
            torch.log((torch.ones(num_heads, 1, 1) / 0.24).exp() - 1))  # Initialize softplus(temperature) to 1/0.24.

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.query_embedding = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(self.num_heads, 1, self.head_dim), mean=0, std=0.02))
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Components to generate pooled features.
        self.pool = nn.AdaptiveAvgPool2d((self.pool_H, self.pool_W))
        self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()

        self.cope = CoPE2D(input_resolution[0]*input_resolution[1], self.head_dim, self.pool_H, self.pool_W)

        # mlp to generate continuous relative position bias
        self.cpb_fc1 = nn.Linear(2, 512, bias=True)
        self.cpb_act = nn.ReLU(inplace=True)
        self.cpb_fc2 = nn.Linear(512, num_heads, bias=True)

        # relative bias for local features
        self.relative_pos_bias_local = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(num_heads, self.local_len), mean=0, std=0.0004))

        # Generate padding_mask && sequnce length scale
        local_seq_length, padding_mask = get_seqlen_and_mask_2d(input_resolution, window_size)
        self.register_buffer("seq_length_scale", torch.as_tensor(np.log(local_seq_length.numpy() + self.pool_len)),
                             persistent=False)
        self.register_buffer("padding_mask", padding_mask, persistent=False)


    def forward(self, x, H, W, relative_pos_index, relative_coords_table):
        B, N, C = x.shape

        q_norm = F.normalize(self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3), dim=-1)
        q_norm_scaled = (q_norm + self.query_embedding) * F.softplus(self.temperature) * self.seq_length_scale

        k_local, v_local = self.kv(x).chunk(2, dim=-1)
        k_local = F.normalize(k_local.reshape(B, N, self.num_heads, self.head_dim), dim=-1).reshape(B, N, -1)

        k_local, v_local = self.unfold(torch.cat([k_local, v_local], dim=-1).permute(0, 2, 1).reshape(B, -1, H, W)).reshape(
            B, 2 * self.num_heads, self.head_dim, self.local_len, N).permute(0, 1, 4, 2, 3).chunk(2, dim=1)

        x_local = ((torch.softmax(((q_norm_scaled.unsqueeze(-2) @ k_local).squeeze(-2) \
                      + self.relative_pos_bias_local.unsqueeze(1)).masked_fill(self.padding_mask, float('-inf')), dim=-1)).unsqueeze(-2)
                   @ v_local.transpose(-2, -1)).squeeze(-2)

        x = x.permute(0, 2, 1).reshape(B, -1, H, W).contiguous()
        x = self.pool(self.act(self.sr(x))).reshape(B, -1, self.pool_len).permute(0, 2, 1)
        x = self.norm(x)

        k_pool, v_pool = (self.kv(x).reshape(B, self.pool_len, 2 * self.num_heads, self.head_dim).permute(0, 2, 1, 3)).chunk(2, dim=1)

        pool_bias = self.cpb_fc2(self.cpb_act(self.cpb_fc1(relative_coords_table))).transpose(0, 1)[:,
                    relative_pos_index.view(-1)].view(-1, N, self.pool_len)
        
        
        x_pool = (torch.softmax(self.cope(q_norm_scaled, q_norm_scaled @ F.normalize(k_pool, dim=-1).transpose(-2, -1) + pool_bias) + 
                                q_norm_scaled @ F.normalize(k_pool, dim=-1).transpose(-2, -1) + pool_bias, dim=-1)) @ v_pool

        x = (x_local + x_pool).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x
    
class Attention(nn.Module):
    def __init__(self, dim, input_resolution, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.temperature = nn.Parameter(
            torch.log((torch.ones(num_heads, 1, 1) / 0.24).exp() - 1))  # Initialize softplus(temperature) to 1/0.24.
        # Generate sequnce length scale
        self.register_buffer("seq_length_scale", torch.as_tensor(np.log(input_resolution[0] * input_resolution[1])),
                             persistent=False)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.query_embedding = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(self.num_heads, 1, self.head_dim), mean=0, std=0.02))

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # mlp to generate continuous relative position bias
        self.cpb_fc1 = nn.Linear(3, 512, bias=True)
        self.cpb_act = nn.ReLU(inplace=True)
        self.cpb_fc2 = nn.Linear(512, num_heads, bias=True)

    def forward(self, x, H, W, T, relative_pos_index, relative_coords_table):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, -1, 3 * self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=1)

        rel_bias = self.cpb_fc2(self.cpb_act(self.cpb_fc1(relative_coords_table))).transpose(0, 1)[:,
                   relative_pos_index.view(-1)].view(-1, N, N)

        attn = ((F.normalize(q, dim=-1) + self.query_embedding) * F.softplus(
            self.temperature) * self.seq_length_scale) @ F.normalize(k, dim=-1).transpose(-2, -1) + rel_bias
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Attention2D(nn.Module):
    def __init__(self, dim, input_resolution, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.temperature = nn.Parameter(
            torch.log((torch.ones(num_heads, 1, 1) / 0.24).exp() - 1))  # Initialize softplus(temperature) to 1/0.24.
        # Generate sequnce length scale
        self.register_buffer("seq_length_scale", torch.as_tensor(np.log(input_resolution[0] * input_resolution[1])),
                             persistent=False)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.query_embedding = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(self.num_heads, 1, self.head_dim), mean=0, std=0.02))

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # mlp to generate continuous relative position bias
        self.cpb_fc1 = nn.Linear(2, 512, bias=True)
        self.cpb_act = nn.ReLU(inplace=True)
        self.cpb_fc2 = nn.Linear(512, num_heads, bias=True)

    def forward(self, x, H, W, relative_pos_index, relative_coords_table):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, -1, 3 * self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=1)

        # Use MLP to generate continuous relative positional bias
        rel_bias = self.cpb_fc2(self.cpb_act(self.cpb_fc1(relative_coords_table))).transpose(0, 1)[:,
                   relative_pos_index.view(-1)].view(-1, N, N)

        # Calculate attention map using sequence length scaled cosine attention and query embedding
        attn = ((F.normalize(q, dim=-1) + self.query_embedding) * F.softplus(
            self.temperature) * self.seq_length_scale) @ F.normalize(k, dim=-1).transpose(-2, -1) + rel_bias
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class CSAttention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # me: support window mask
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = attn.softmax(dim=-1)
        else:
            attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

