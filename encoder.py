import torch.nn as nn
import torch
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from positon import get_relative_position_cpb, get_relative_position_cpb_2D
from attention import AggregatedAttention, Attention, AggregatedAttention2D, Attention2D, CSAttention
import torch.nn.functional as F

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, patch_size=(16, 16, 2), in_chans=3, embed_dim=768):
        super().__init__()

        self.proj = nn.Conv3d(in_channels=in_chans, out_channels=embed_dim, 
                            kernel_size = (patch_size[2],  patch_size[0], patch_size[1]), 
                            stride=(patch_size[2],  patch_size[0], patch_size[1]))        
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, **kwargs):
        x = self.proj(x)
        B, C, T, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        return x, H, W, T
    
class PatchEmbed2D(nn.Module):
    """ Flexible Image to Patch Embedding
    """

    def __init__(self, patch_size=(16, 16), in_chans=3, embed_dim=768):
        super().__init__()

        self.proj = nn.Conv2d(in_channels=in_chans, out_channels=embed_dim, 
                            kernel_size = (patch_size[0], patch_size[1]), 
                            stride=(patch_size[0], patch_size[1]))        
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, **kwargs):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        return x, H, W

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement 
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, input_resolution, window_size=3, mlp_ratio=4.,
                 qkv_bias=False, drop=0., attn_drop=0., 
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, t_ratio=1, fixed_pool_size=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if sr_ratio == 1:
            self.attn = Attention(
                dim,
                input_resolution,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop)
        else:
            self.attn = AggregatedAttention(
                dim,
                input_resolution,
                window_size=window_size,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
                sr_ratio=sr_ratio,
                t_ratio=t_ratio,
                fixed_pool_size=fixed_pool_size)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, H, W, T, relative_pos_index, relative_coords_table):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W, T, relative_pos_index, relative_coords_table))
        # x = x + self.drop_path(self.mlp(self.norm2(x), H, W, T))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
class Block2D(nn.Module):

    def __init__(self, dim, num_heads, input_resolution, window_size=3, mlp_ratio=4.,
                 qkv_bias=False, drop=0., attn_drop=0., 
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, t_ratio=1, fixed_pool_size=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if sr_ratio == 1:
            self.attn = Attention2D(
                dim,
                input_resolution,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop)
        else:
            self.attn = AggregatedAttention2D(
                dim,
                input_resolution,
                window_size=window_size,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
                sr_ratio=sr_ratio,
                fixed_pool_size=fixed_pool_size)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, H, W, relative_pos_index, relative_coords_table):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W, relative_pos_index, relative_coords_table))
        # x = x + self.drop_path(self.mlp(self.norm2(x), H, W, T))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# class PretrainVisionTransformerEncoder(nn.Module):
#     """ Vision Transformer with support for patch or hybrid CNN input stage
#     """
#     def __init__(self, img_size=224, patch_size=16, frame_nums=16, tubelet_size=2, in_chans=3, num_classes=0, 
#                  embed_dims=[64, 128, 256, 512], window_size=[3, 3, 3, None], 
#                  num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], num_stages=4, depths=[3, 4, 6, 3],
#                  qkv_bias=False, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, 
#                  sr_ratios=[8, 4, 2, 1], t_ratios=[2, 2, 2, 1], fixed_pool_size=None
#                  ):
#         super().__init__()
#         self.num_classes = num_classes
#         self.depths = depths
#         self.num_stages = num_stages

#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
#         cur = 0

#         for i in range(num_stages):
#             # Generate relative positional coordinate table and index for each stage to compute continuous relative positional bias.
            
#             query_size=(img_size // (patch_size ** (i + 1)), 
#                         img_size // (patch_size ** (i + 1)), 
#                         frame_nums // (tubelet_size ** (i + 1)))
#             key_size=(img_size // ((patch_size ** (i + 1)) * sr_ratios[i]), 
#                       img_size // ((patch_size ** (i + 1)) * sr_ratios[i]), 
#                       frame_nums // ((tubelet_size ** (i + 1)) * t_ratios[i])) \
#                           if (fixed_pool_size is None or sr_ratios[i] == 1) else fixed_pool_size
#             pretrain_size=(img_size // (2 ** (i + 2)), img_size // (2 ** (i + 2)), frame_nums // (2 ** (i + 1)))
            
#             relative_pos_index, relative_coords_table = get_relative_position_cpb(
#                 query_size=query_size,
#                 key_size=key_size,
#                 pretrain_size=pretrain_size
#             )

#             self.register_buffer(f"relative_pos_index{i + 1}", relative_pos_index, persistent=False)
#             self.register_buffer(f"relative_coords_table{i + 1}", relative_coords_table, persistent=False)

#             # patch_embed = PatchEmbed(patch_size=(patch_size ** (i + 1), patch_size ** (i + 1), tubelet_size ** (i + 1)),
#             #                         in_chans=in_chans if i == 0 else embed_dims[i - 1],
#             #                         embed_dim=embed_dims[i])
            
#             patch_embed = PatchEmbed(patch_size=(patch_size, patch_size, tubelet_size),
#                                     in_chans=in_chans if i == 0 else embed_dims[i - 1],
#                                     embed_dim=embed_dims[i])

#             block = nn.ModuleList([Block(
#                 dim=embed_dims[i], input_resolution=query_size, window_size=window_size[i],
#                 num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
#                 drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
#                 sr_ratio=sr_ratios[i], t_ratio=t_ratios[i], fixed_pool_size=fixed_pool_size)
#                 for j in range(depths[i])])
#             norm = norm_layer(embed_dims[i])
#             cur += depths[i]

#             setattr(self, f"patch_embed{i + 1}", patch_embed)
#             setattr(self, f"block{i + 1}", block)
#             setattr(self, f"norm{i + 1}", norm)
            
#         self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
#         self.apply(self._init_weights)


#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             nn.init.xavier_uniform_(m.weight)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)

#     def get_num_layers(self):
#         return len(self.blocks)

#     @torch.jit.ignore
#     def no_weight_decay(self):
#         return {'pos_embed', 'cls_token', 'part_tokens'}

#     def get_classifier(self):
#         return self.head

#     def reset_classifier(self, num_classes, global_pool=''):
#         self.num_classes = num_classes
#         self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

#     def forward_features(self, x, mask, return_intermediate_features=None):
#         intermediate_features = []
        
#         for i in range(self.num_stages):
#             patch_embed = getattr(self, f"patch_embed{i + 1}")
#             block = getattr(self, f"block{i + 1}")
#             norm = getattr(self, f"norm{i + 1}")
#             x, H, W, T = patch_embed(x)
#             if i == 0:
#                 B, _, C = x.shape
#                 x = x[~mask].reshape(B, -1, C) # ~mask means visible
#             relative_pos_index = getattr(self, f"relative_pos_index{i + 1}")
#             relative_coords_table = getattr(self, f"relative_coords_table{i + 1}")
#             for blk in block:
#                 x = blk(x, H, W, T, relative_pos_index, relative_coords_table)
#             x = norm(x)
#             intermediate_features.append(x)
#             if i != self.num_stages - 1:
#                 x = x.reshape(B, H, W, T, -1).permute(0, 4, 3, 1, 2).contiguous()

#         if return_intermediate_features is None:
#             return x, None
#         else:
#             intermediate_features = [intermediate_features[i] for i in return_intermediate_features]
#             return x, intermediate_features

#     def forward(self, x, mask=None, return_intermediate_features=None):
#         x, intermediate_features = self.forward_features(x, mask, return_intermediate_features)
#         x = self.head(x)
#         return x, intermediate_features
    
# class PretrainVisionTransformerEncoder2D(nn.Module):
#     """ Vision Transformer with support for patch or hybrid CNN input stage
#     """

#     def __init__(self, img_size_audio=(256, 128), patch_size=16, in_chans=3, num_classes=0, 
#                  embed_dims=[64, 128, 256, 512], depths=[3, 4, 6, 3], num_heads=12, 
#                  sr_ratios=[8, 4, 2, 1], qkv_bias=False, qk_scale=None, drop_rate=0., 
#                  attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, 
#                  use_checkpoint=False, num_stages=4, window_size=[3, 3, 3, None],
#                  use_learnable_pos_emb=False, mlp_ratios=[4, 4, 4, 4], fixed_pool_size=None):
#         super().__init__()
#         self.num_classes = num_classes
#         self.depths = depths
#         self.num_stages = num_stages
        
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
#         cur = 0
        
#         for i in range(num_stages):
#             # Generate relative positional coordinate table and index for each stage to compute continuous relative positional bias.
            
#             query_size=(img_size_audio[0] // (patch_size ** (i + 1)), 
#                         img_size_audio[1] // (patch_size ** (i + 1)))
#             key_size=(img_size_audio[0] // ((patch_size ** (i + 1)) * sr_ratios[i]), 
#                       img_size_audio[1] // ((patch_size ** (i + 1)) * sr_ratios[i]))
            
#             pretrain_size=(img_size_audio[0] // (2 ** (i + 2)), img_size_audio[1] // (2 ** (i + 2)))
#             relative_pos_index, relative_coords_table = get_relative_position_cpb(
#                 query_size=query_size,
#                 key_size=key_size,
#                 pretrain_size=pretrain_size
#             )

#             self.register_buffer(f"relative_pos_index{i + 1}", relative_pos_index, persistent=False)
#             self.register_buffer(f"relative_coords_table{i + 1}", relative_coords_table, persistent=False)

#             # patch_embed = PatchEmbed(patch_size=(patch_size ** (i + 1), patch_size ** (i + 1), tubelet_size ** (i + 1)),
#             #                         in_chans=in_chans if i == 0 else embed_dims[i - 1],
#             #                         embed_dim=embed_dims[i])
            
#             patch_embed = PatchEmbed2D(img_size=img_size_audio, patch_size=patch_size,
#                                     in_chans=in_chans if i == 0 else embed_dims[i - 1],
#                                     embed_dim=embed_dims[i], stride=patch_size)

#             block = nn.ModuleList([Block(
#                 dim=embed_dims[i], input_resolution=query_size, window_size=window_size[i],
#                 num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
#                 drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
#                 sr_ratio=sr_ratios[i], fixed_pool_size=fixed_pool_size)
#                 for j in range(depths[i])])
#             norm = norm_layer(embed_dims[i])
#             cur += depths[i]

#             setattr(self, f"patch_embed{i + 1}", patch_embed)
#             setattr(self, f"block{i + 1}", block)
#             setattr(self, f"norm{i + 1}", norm)
            
#         self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()

#         if use_learnable_pos_emb:
#             trunc_normal_(self.pos_embed, std=.02)

#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             nn.init.xavier_uniform_(m.weight)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)

#     def get_num_layers(self):
#         return len(self.blocks)

#     @torch.jit.ignore
#     def no_weight_decay(self):
#         return {'pos_embed', 'cls_token'}

#     def get_classifier(self):
#         return self.head

#     def reset_classifier(self, num_classes, global_pool=''):
#         self.num_classes = num_classes
#         self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

#     def forward_features(self, x, mask, return_intermediate_features=None):
#         x = self.patch_embed(x)

#         x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()

#         B, _, C = x.shape
#         x_vis = x[~mask].reshape(B, -1, C)  # ~mask means visible

#         intermediate_features = []

#         for blk in self.blocks:
#             x_vis = blk(x_vis)
#             intermediate_features.append(self.norm(x_vis))

#         x_vis = self.norm(x_vis)
        
#         if return_intermediate_features is None:
#             return x_vis, None
#         else:
#             intermediate_features = [intermediate_features[i] for i in return_intermediate_features]
#             return x_vis, intermediate_features

#     def forward(self, x, mask, return_intermediate_features=None):
#         x, intermediate_features = self.forward_features(x, mask, return_intermediate_features)
#         x = self.head(x)
#         return x, intermediate_features

class GeneralAttention(nn.Module):
    def __init__(
            self, dim, context_dim=None, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, all_head_dim, bias=False)
        self.kv = nn.Linear(dim if context_dim is None else context_dim, all_head_dim * 2, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context=None):
        B, T1, C = x.shape
        q_bias, kv_bias = self.q_bias, None
        if self.q_bias is not None:
            kv_bias = torch.cat((torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        q = F.linear(input=x, weight=self.q.weight, bias=q_bias)
        q = q.reshape(B, T1, self.num_heads, -1).transpose(1,2) # me: (B, H, T1, C//H)
        kv = F.linear(input=x if context is None else context, weight=self.kv.weight, bias=kv_bias)
        _, T2, _ = kv.shape
        kv = kv.reshape(B, T2, 2, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1] # make torchscript happy (cannot use tensor as tuple), me： (B, H, T2, C//H)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)) # me: (B, H, T1, T2)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, T1, -1) # (B, H, T1, C//H) -> (B, T1, H, C//H) -> (B, T1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CSBlock(nn.Module):
    def __init__(self, dim, context_dim, num_heads, num_cross_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None, cross_attn_head_dim=None):
        super().__init__()

        self.cross_attn = GeneralAttention(
            dim=dim, context_dim=context_dim, num_heads=num_cross_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=cross_attn_head_dim)
        self.cross_norm1 = norm_layer(dim)
        self.cross_norm2 = norm_layer(context_dim)

        self.norm1 = norm_layer(dim)
        self.attn = CSAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_0 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_0, self.gamma_1, self.gamma_2 = None, None, None

    def forward(self, x, context):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.cross_attn(self.cross_norm1(x), context=self.cross_norm2(context)))
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_0 * self.cross_attn(self.cross_norm1(x), context=self.cross_norm2(context)))
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PretrainVisionTransformerEncoderForFusion(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, embed_dim=768, embed_dim_audio=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
                 modal_param_sharing=False):
        super().__init__()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            CSBlock(dim=embed_dim, context_dim=embed_dim_audio, num_heads=num_heads,
                    num_cross_heads=num_heads,
                    mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=dpr[i], norm_layer=norm_layer,
                    init_values=init_values)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)

        # audio
        self.modal_param_sharing = modal_param_sharing
        if not modal_param_sharing:
            self.blocks_audio = nn.ModuleList([
                CSBlock(dim=embed_dim_audio, context_dim=embed_dim, num_heads=num_heads,
                        num_cross_heads=num_heads,
                        mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[i], norm_layer=norm_layer,
                        init_values=init_values)
                for i in range(depth)])
        else:
            self.blocks_audio = self.blocks

        # do not share norm layer
        self.norm_audio = norm_layer(embed_dim_audio)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, x_audio):
        for blk, blk_audio in zip(self.blocks, self.blocks_audio):
            x, x_audio = blk(x, context=x_audio), blk_audio(x_audio, context=x)

        # norm
        x = self.norm(x)

        x_audio = self.norm_audio(x_audio)

        return x, x_audio
    
class PretrainResidualTransformerEncoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, img_size_audio=(256, 128), frame_nums=16, tubelet_size=2, 
                 in_chans=3, num_classes=0, embed_dims=[64, 128, 256, 512], window_size=[3, 3, 3, None], 
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], num_stages=4, depths=[3, 4, 6, 3],
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, 
                 sr_ratios=[8, 4, 2, 1], t_ratios=[2, 2, 2, 1], fixed_pool_size=None, init_values=0.
                 ):
        super().__init__()
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        
        cur = 0
        for i in range(num_stages):
            # Generate relative positional coordinate table and index for each stage to compute continuous relative positional bias.
            
            query_size=(img_size // (patch_size ** (i + 1)), 
                        img_size // (patch_size ** (i + 1)), 
                        frame_nums // (tubelet_size ** (i + 1)))
            key_size=(img_size // ((patch_size ** (i + 1)) * sr_ratios[i]), 
                      img_size // ((patch_size ** (i + 1)) * sr_ratios[i]), 
                      frame_nums // ((tubelet_size ** (i + 1)) * t_ratios[i])) \
                          if (fixed_pool_size is None or sr_ratios[i] == 1) else fixed_pool_size
            # pretrain_size=(img_size // (2 ** (i + 2)), img_size // (2 ** (i + 2)), frame_nums // (2 ** (i + 1)))
            
            relative_pos_index, relative_coords_table = get_relative_position_cpb(
                query_size=query_size,
                key_size=key_size,
                # pretrain_size=pretrain_size
            )

            self.register_buffer(f"vi_relative_pos_index{i + 1}", relative_pos_index, persistent=False)
            self.register_buffer(f"vi_relative_coords_table{i + 1}", relative_coords_table, persistent=False)

            # patch_embed = PatchEmbed(patch_size=(patch_size ** (i + 1), patch_size ** (i + 1), tubelet_size ** (i + 1)),
            #                         in_chans=in_chans if i == 0 else embed_dims[i - 1],
            #                         embed_dim=embed_dims[i])
            
            patch_embed = PatchEmbed(patch_size=(patch_size, patch_size, tubelet_size),
                                    in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                    embed_dim=embed_dims[i])

            block = nn.ModuleList([Block(
                dim=embed_dims[i], input_resolution=query_size, window_size=window_size[i],
                num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i], t_ratio=t_ratios[i], fixed_pool_size=fixed_pool_size)
                for j in range(depths[i])])
            
            csblock = nn.ModuleList([CSBlock(
                dim=embed_dims[i], context_dim=embed_dims[i], num_heads=num_heads[i],
                num_cross_heads=num_heads[i],
                mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[cur + j], norm_layer=norm_layer,
                init_values=init_values)
                for j in range(depths[i])])
            
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"vi_patch_embed{i + 1}", patch_embed)
            setattr(self, f"vi_block{i + 1}", block)
            setattr(self, f"vi_csblock{i + 1}", csblock)
            setattr(self, f"vi_norm{i + 1}", norm)
            
            
        cur = 0
        for i in range(num_stages):
            # Generate relative positional coordinate table and index for each stage to compute continuous relative positional bias.
            
            query_size=(img_size_audio[0] // (patch_size ** (i + 1)), 
                        img_size_audio[1] // (patch_size ** (i + 1)))
            key_size=(img_size_audio[0] // ((patch_size ** (i + 1)) * sr_ratios[i]), 
                      img_size_audio[1] // ((patch_size ** (i + 1)) * sr_ratios[i]))
            
            # pretrain_size=(img_size_audio[0] // (2 ** (i + 2)), img_size_audio[1] // (2 ** (i + 2)))
            relative_pos_index, relative_coords_table = get_relative_position_cpb_2D(
                query_size=query_size,
                key_size=key_size,
                # pretrain_size=pretrain_size
            )

            self.register_buffer(f"au_relative_pos_index{i + 1}", relative_pos_index, persistent=False)
            self.register_buffer(f"au_relative_coords_table{i + 1}", relative_coords_table, persistent=False)

            # patch_embed = PatchEmbed(patch_size=(patch_size ** (i + 1), patch_size ** (i + 1), tubelet_size ** (i + 1)),
            #                         in_chans=in_chans if i == 0 else embed_dims[i - 1],
            #                         embed_dim=embed_dims[i])
            
            patch_embed = PatchEmbed2D(patch_size=(patch_size, patch_size), 
                                       in_chans=1 if i == 0 else embed_dims[i - 1], 
                                       embed_dim=embed_dims[i] )

            block = nn.ModuleList([Block2D(
                dim=embed_dims[i], input_resolution=query_size, window_size=window_size[i],
                num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i], fixed_pool_size=fixed_pool_size)
                for j in range(depths[i])])
            
            csblock = nn.ModuleList([CSBlock(
                dim=embed_dims[i], context_dim=embed_dims[i], num_heads=num_heads[i],
                num_cross_heads=num_heads[i],
                mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[cur + j], norm_layer=norm_layer,
                init_values=init_values)
                for j in range(depths[i])])
            
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"au_patch_embed{i + 1}", patch_embed)
            setattr(self, f"au_block{i + 1}", block)
            setattr(self, f"au_csblock{i + 1}", csblock)
            setattr(self, f"au_norm{i + 1}", norm)
            
        self.head = nn.Linear(embed_dims[-1]*2, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.vi_block1)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'part_tokens'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x_vi, mask_vi, x_au, mask_au, return_intermediate_features=None):
        intermediate_features_vi = []
        intermediate_features_au = []
        
        for i in range(self.num_stages):
            vi_patch_embed = getattr(self, f"vi_patch_embed{i + 1}")
            vi_block = getattr(self, f"vi_block{i + 1}")
            vi_csblock = getattr(self, f"vi_csblock{i + 1}")
            vi_norm = getattr(self, f"vi_norm{i + 1}")
            x_vi, H_vi, W_vi, T_vi = vi_patch_embed(x_vi)
            
            if i == 0 and mask_vi != None:
                B, _, C = x_vi.shape
                # x_vi = x_vi[~mask_vi].reshape(B, -1, C) # ~mask means visible
                x_vi[mask_vi] = 0
            x_vi = vi_norm(x_vi)
            vi_relative_pos_index = getattr(self, f"vi_relative_pos_index{i + 1}")
            vi_relative_coords_table = getattr(self, f"vi_relative_coords_table{i + 1}")
            
            au_patch_embed = getattr(self, f"au_patch_embed{i + 1}")
            au_block = getattr(self, f"au_block{i + 1}")
            au_csblock = getattr(self, f"au_csblock{i + 1}")
            au_norm = getattr(self, f"au_norm{i + 1}")
            x_au, H_au, W_au = au_patch_embed(x_au)
            
            if i == 0 and mask_au != None:
                B, _, C = x_au.shape
                # x_au = x_au[~mask_au].reshape(B, -1, C) # ~mask means visible
                x_au[mask_au] = 0
            x_au = au_norm(x_au)

            au_relative_pos_index = getattr(self, f"au_relative_pos_index{i + 1}")
            au_relative_coords_table = getattr(self, f"au_relative_coords_table{i + 1}")
            
            for vi_blk, au_blk in zip(vi_block, au_block):
                x_vi = vi_blk(x_vi, H_vi, W_vi, T_vi, vi_relative_pos_index, vi_relative_coords_table)
                x_au = au_blk(x_au, H_au, W_au, au_relative_pos_index, au_relative_coords_table)
            
            x_vi = vi_norm(x_vi)
            x_au = au_norm(x_au)
            intermediate_features_vi.append(x_vi)
            intermediate_features_au.append(x_au)
            
            for vi_blk, au_blk in zip(vi_csblock, au_csblock):
                x_vi, x_au = vi_blk(x_vi, context=x_au), au_blk(x_au, context=x_vi)

            # norm
            x_vi = vi_norm(x_vi)
            x_au = au_norm(x_au)
            intermediate_features_vi.append(x_vi)
            intermediate_features_au.append(x_au)
            
            if i != self.num_stages - 1:
                x_vi = x_vi.reshape(B, H_vi, W_vi, T_vi, -1).permute(0, 4, 3, 1, 2).contiguous()
                x_au = x_au.reshape(B, H_au, W_au, -1).permute(0, 3, 1, 2).contiguous()

        if return_intermediate_features is None:
            return x_vi, x_au, None, None
        else:
            # intermediate_features_vi = [intermediate_features_vi[i] for i in return_intermediate_features]
            # intermediate_features_au = [intermediate_features_au[i] for i in return_intermediate_features]
            return x_vi, x_au, intermediate_features_vi, intermediate_features_au

    def forward(self, x_vi, x_au, mask_vi=None, mask_au=None, return_intermediate_features=None):
        x_vi, x_au, intermediate_features_vi, intermediate_features_au = self.forward_features(x_vi, mask_vi, x_au, mask_au, return_intermediate_features)
        # x = self.head(x)
        return x_vi, x_au, intermediate_features_vi, intermediate_features_au

class PretrainVisionTransformerDecoder(nn.Module):
    def __init__(self, img_size=224, patch_size=4, frame_nums=16, fixed_pool_size=None,
                 depths=[8, 8], num_stages=2, embed_dims=[128, 64], window_size=[3, 3, 3, None], mlp_ratios=[4, 4, 4, 4],
                 num_heads=[8, 4], qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., sr_ratios=[4, 8],
                 t_ratios=[2, 2], drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, tubelet_size=2
                 ):
        super().__init__()
        
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.depths = depths
        self.embed_dims = embed_dims
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        
        cur = 0
        for i in range(num_stages):
            # Generate relative positional coordinate table and index for each stage to compute continuous relative positional bias.
            
            query_size=(img_size // (patch_size ** (num_stages-i)), 
                        img_size // (patch_size ** (num_stages-i)), 
                        frame_nums // (tubelet_size ** (num_stages-i)))
            key_size=(img_size // ((patch_size ** (num_stages-i)) * sr_ratios[i]), 
                      img_size // ((patch_size ** (num_stages-i)) * sr_ratios[i]), 
                      frame_nums // ((tubelet_size ** (num_stages-i)) * t_ratios[i]))
            
            pretrain_size=(img_size // (2 ** (num_stages+1-i)), img_size // (2 ** (num_stages+1-i)), frame_nums // (2 ** (num_stages-i)))
            
            relative_pos_index, relative_coords_table = get_relative_position_cpb(
                query_size=query_size,
                key_size=key_size,
                pretrain_size=pretrain_size
            )
            
            if(i == num_stages-1):
                num_classes = 3 * tubelet_size * patch_size ** 2 
            else:
                num_classes = embed_dims[num_stages-1-i] * tubelet_size * patch_size ** 2 

            self.register_buffer(f"relative_pos_index{i + 1}", relative_pos_index, persistent=False)
            self.register_buffer(f"relative_coords_table{i + 1}", relative_coords_table, persistent=False)

            # patch_embed = PatchEmbed(patch_size=(patch_size ** (i + 1), patch_size ** (i + 1), tubelet_size ** (i + 1)),
            #                         in_chans=in_chans if i == 0 else embed_dims[i - 1],
            #                         embed_dim=embed_dims[i])
            
            block = nn.ModuleList([Block(
                dim=embed_dims[i], input_resolution=query_size, window_size=window_size[i],
                num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i], t_ratio=t_ratios[i], fixed_pool_size=fixed_pool_size)
                for j in range(depths[i])])
            
            norm = norm_layer(embed_dims[i])
            head = nn.Linear(embed_dims[i], num_classes)
            cur += depths[i]

            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)
            setattr(self, f"head{i + 1}", head)
            setattr(self, f"H{i + 1}", query_size[0])
            setattr(self, f"W{i + 1}", query_size[1])
            setattr(self, f"T{i + 1}", query_size[2])


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, return_intermediate_features=None):
        intermediate_features = []
        B, _, _ = x.shape
        
        for i in range(self.num_stages):
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            head = getattr(self, f"head{i + 1}")
            H = getattr(self, f"H{i + 1}")
            W = getattr(self, f"W{i + 1}")
            T = getattr(self, f"T{i + 1}")
            
            relative_pos_index = getattr(self, f"relative_pos_index{i + 1}")
            relative_coords_table = getattr(self, f"relative_coords_table{i + 1}")
            
            for blk in block:
                x = blk(x, H, W, T, relative_pos_index, relative_coords_table)
                
            x = head(norm(x))
            
            intermediate_features.append(x)
            
            if i == self.num_stages-1:
                x = x.reshape(B, H*self.patch_size, W*self.patch_size, T*self.tubelet_size, -1).permute(0, 4, 3, 1, 2).contiguous()
            else:
                x = x.reshape(B, -1, self.embed_dims[i+1])

        if return_intermediate_features is None:
            return x, None
        else:
            return x, intermediate_features

    def forward(self, x, return_intermediate_features=None):
        x, intermediate_features = self.forward_features(x, return_intermediate_features)
        return x, intermediate_features
    
class PretrainVisionTransformerDecoder2D(nn.Module):
    def __init__(self, img_size_audio=(256, 128), patch_size=4, fixed_pool_size=None,
                 depths=[8, 8], num_stages=2, embed_dims=[128, 64], window_size=[3, 3, 3, None], mlp_ratios=[4, 4, 4, 4],
                 num_heads=[8, 4], qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., sr_ratios=[4, 8],
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None
                 ):
        super().__init__()
        
        self.patch_size = patch_size
        self.depths = depths
        self.embed_dims = embed_dims
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        
        cur = 0
        for i in range(num_stages):
            # Generate relative positional coordinate table and index for each stage to compute continuous relative positional bias.
            
            query_size=(img_size_audio[0] // (patch_size ** (num_stages-i)), 
                        img_size_audio[1] // (patch_size ** (num_stages-i)))
            key_size=(img_size_audio[0] // ((patch_size ** (num_stages-i)) * sr_ratios[i]), 
                      img_size_audio[1] // ((patch_size ** (num_stages-i)) * sr_ratios[i]))
            
            pretrain_size=(img_size_audio[0] // (2 ** (num_stages+1-i)), img_size_audio[1] // (2 ** (num_stages+1-i)))
            
            relative_pos_index, relative_coords_table = get_relative_position_cpb_2D(
                query_size=query_size,
                key_size=key_size,
                pretrain_size=pretrain_size
            )
            
            if(i == num_stages-1):
                num_classes = 1 * patch_size ** 2 
            else:
                num_classes = embed_dims[num_stages-1-i] * patch_size ** 2 

            self.register_buffer(f"relative_pos_index{i + 1}", relative_pos_index, persistent=False)
            self.register_buffer(f"relative_coords_table{i + 1}", relative_coords_table, persistent=False)

            # patch_embed = PatchEmbed(patch_size=(patch_size ** (i + 1), patch_size ** (i + 1), tubelet_size ** (i + 1)),
            #                         in_chans=in_chans if i == 0 else embed_dims[i - 1],
            #                         embed_dim=embed_dims[i])
            
            block = nn.ModuleList([Block2D(
                dim=embed_dims[i], input_resolution=query_size, window_size=window_size[i],
                num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i], fixed_pool_size=fixed_pool_size)
                for j in range(depths[i])])
            
            norm = norm_layer(embed_dims[i])
            head = nn.Linear(embed_dims[i], num_classes)
            cur += depths[i]

            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)
            setattr(self, f"head{i + 1}", head)
            setattr(self, f"H{i + 1}", query_size[0])
            setattr(self, f"W{i + 1}", query_size[1])


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, return_intermediate_features=None):
        intermediate_features = []
        B, _, _ = x.shape
        
        for i in range(self.num_stages):
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            head = getattr(self, f"head{i + 1}")
            H = getattr(self, f"H{i + 1}")
            W = getattr(self, f"W{i + 1}")
            
            relative_pos_index = getattr(self, f"relative_pos_index{i + 1}")
            relative_coords_table = getattr(self, f"relative_coords_table{i + 1}")
            
            for blk in block:
                x = blk(x, H, W, relative_pos_index, relative_coords_table)
                
            x = head(norm(x))
            
            intermediate_features.append(x)
            
            if i == self.num_stages-1:
                x = x.reshape(B, H*self.patch_size, W*self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
            else:
                x = x.reshape(B, -1, self.embed_dims[i+1])

        if return_intermediate_features is None:
            return x, None
        else:
            return x, intermediate_features

    def forward(self, x, return_intermediate_features=None):
        x, intermediate_features = self.forward_features(x, return_intermediate_features)
        return x, intermediate_features

if __name__ == "__main__":
    # model = PatchEmbed()
    # x = torch.rand((2, 3, 16, 224, 224))
    # # x = model(x)
    # visionmodel = PretrainVisionTransformerEncoder(img_size=224, patch_size=4, frame_nums=16, tubelet_size=2, in_chans=3, num_classes=10, 
    #              embed_dims=[64, 128, 256, 512], window_size=[3, 3, 3, None], 
    #              num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], num_stages=2, depths=[4, 4],
    #              qkv_bias=False, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, 
    #              sr_ratios=[8, 4, 2, 1], t_ratios=[2, 2, 2, 1], fixed_pool_size=None)
    # x = visionmodel(x)
    # print()
    
    
    
    
    
    
    import random
    
    vi = torch.rand((2, 3, 16, 224, 224))
    au = torch.rand((2, 1, 256, 128))
    
    encoder = PretrainResidualTransformerEncoder(img_size=224, patch_size=4, img_size_audio=(256, 128), frame_nums=16, tubelet_size=2, 
                 in_chans=3, num_classes=0, embed_dims=[64, 512], window_size=[3, 3], 
                 num_heads=[4, 8], mlp_ratios=[4, 4], num_stages=2, depths=[8, 8],
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, 
                 sr_ratios=[8, 4], t_ratios=[2, 2], fixed_pool_size=None, init_values=0.)
    
    # 1568
    # mask_vi = [random.choice([False, True]) for _ in range(1568)]
    # 128
    # mask_au = [random.choice([False, True]) for _ in range(128)]
    
    x_vi, x_au, in_vi, in_au = encoder(vi, au, return_intermediate_features=True)
    
    
    
    
    
    x_vi = torch.rand((2, 784, 512))
    
    decoder = PretrainVisionTransformerDecoder(img_size=224, patch_size=4, frame_nums=16, fixed_pool_size=None,
                 depths=[8, 8], num_stages=2, embed_dims=[512, 64], window_size=[3, 3], mlp_ratios=[4, 4],
                 num_heads=[8, 4], qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., sr_ratios=[4, 8],
                 t_ratios=[2, 2], drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, tubelet_size=2
                 )
    
    x, _ = decoder(x_vi)
    
    
    x_au = torch.rand((2, 128, 512))
    
    decoder = PretrainVisionTransformerDecoder2D(img_size_audio=(256, 128), patch_size=4, fixed_pool_size=None,
                 depths=[8, 8], num_stages=2, embed_dims=[512, 64], window_size=[3, 3, 3, None], mlp_ratios=[4, 4, 4, 4],
                 num_heads=[8, 4], qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., sr_ratios=[4, 8],
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None
                 )
    
    x, _ = decoder(x_au)
    
    
    print()
    