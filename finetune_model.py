from timm.models.registry import register_model
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import PretrainResidualTransformerEncoder, PretrainVisionTransformerDecoder, PretrainVisionTransformerDecoder2D
from functools import partial
from timm.models import create_model

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 400, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }

class AudioVisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self,
                 img_size=224, 
                 patch_size=4, 
                 img_size_audio=(256, 128), 
                 frame_nums=16, 
                 tubelet_size=2, 
                 in_chans=3, 
                 num_classes=1000, 
                 embed_dims=[64, 512], 
                 window_size=[3, 3], 
                 num_heads=[4, 8], 
                 mlp_ratios=[4, 4], 
                 num_stages=2, 
                 depths=[8, 8],
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0., 
                 drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, 
                 sr_ratios=[8, 4], 
                 t_ratios=[2, 2], 
                 fixed_pool_size=None, 
                 init_values=0.,
                 head_activation_func=None
                 ):
        super().__init__()

        # encoder
        self.encoder = PretrainResidualTransformerEncoder(
            img_size=img_size, 
            patch_size=patch_size, 
            img_size_audio=img_size_audio, 
            frame_nums=frame_nums, 
            tubelet_size=tubelet_size, 
            in_chans=in_chans, 
            num_classes=0, 
            embed_dims=embed_dims, 
            window_size=window_size, 
            num_heads=num_heads, 
            mlp_ratios=mlp_ratios, 
            num_stages=num_stages, 
            depths=depths,
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate, 
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            sr_ratios=sr_ratios, 
            t_ratios=t_ratios, 
            fixed_pool_size=fixed_pool_size, 
            init_values=init_values)
        
        fc_num = 4 * embed_dims[-1]
        self.fc_norm = norm_layer(fc_num)
        self.head = nn.Linear(fc_num, num_classes) if num_classes > 0 else nn.Identity()

        if head_activation_func is not None:
            if head_activation_func == 'sigmoid':
                self.head_activation_func = nn.Sigmoid()
            elif head_activation_func == 'relu':
                self.head_activation_func = nn.ReLU()
            elif head_activation_func == 'tanh':
                self.head_activation_func = nn.Tanh()
            else:
                raise NotImplementedError
        else: # default
            self.head_activation_func = nn.Identity()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.encoder.vi_block1)
    
    def get_num_modality_specific_layers(self):
        # return len(self.blocks)
        return len(self.encoder.vi_block1)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'encoder.vi_patch_embed', 'encoder.au_patch_embed'}

    def forward(self, x, x_audio, save_feature=False):
        
        x_vi, x_au, x_vis_inter_features, x_vis_audio_inter_features = self.encoder(x, x_audio, None, None, return_intermediate_features=True)
        
        fusion_video_feature = x_vi.mean(dim=1)  # [B, L, C] -> [B, C]
        fusion_audio_feature = x_au.mean(dim=1)
        
        video_feature = x_vis_inter_features[-1].mean(dim=1)
        audio_feature = x_vis_audio_inter_features[-1].mean(dim=1)
        
        final_feature = torch.cat([video_feature, audio_feature,
                                   fusion_video_feature, fusion_audio_feature],
                                   dim=-1)
        final_feature = self.fc_norm(final_feature)
        
        if save_feature:
            feature = final_feature

        x = self.head(final_feature)

        x = self.head_activation_func(x)

        if save_feature:
            return x, feature
        else:
            return x # (B, C)

@register_model
def avit_dim512_patch16_160_a256(pretrained=False, **kwargs):
    embed_dim = 512
    num_heads = 8
    patch_size = 16
    model = AudioVisionTransformer(
        img_size=160, 
        patch_size=patch_size, 
        img_size_audio=(256, 128), 
        tubelet_size=2, 
        in_chans=3, 
        embed_dims=[embed_dim], 
        window_size=[3], 
        num_heads=[num_heads], 
        mlp_ratios=[4], 
        num_stages=1, 
        depths=[16],
        qkv_bias=False, 
        qk_scale=None, 
        drop_rate=0., 
        attn_drop_rate=0., 
        drop_path_rate=0., 
        norm_layer=nn.LayerNorm, 
        sr_ratios=[8], 
        t_ratios=[2], 
        fixed_pool_size=None, 
        init_values=0., 
        **kwargs)
    model.default_cfg = _cfg()
    return model

if __name__ == "__main__":
    model = create_model(
        "avit_dim512_patch16_160_a256",
        pretrained=False,
        num_classes=7
    )
    
    vi = torch.rand((1, 3, 16, 160, 160))
    au = torch.rand((1, 1, 256, 128))
    
    re = model(vi, au)
    print()
    