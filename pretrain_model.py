from timm.models.registry import register_model
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import PretrainResidualTransformerEncoder, PretrainVisionTransformerDecoder, PretrainVisionTransformerDecoder2D
from functools import partial

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 400, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }

class PretrainAudioVisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self,
                 img_size=224, 
                 patch_size=4, 
                 img_size_audio=(256, 128), 
                 frame_nums=16, 
                 tubelet_size=2, 
                 in_chans=3, 
                 num_classes=0, 
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
                 inter_contrastive_temperature=0.07
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
            num_classes=num_classes, 
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

        # decoder
        self.decoder = PretrainVisionTransformerDecoder(
            img_size=img_size, 
            patch_size=patch_size, 
            frame_nums=frame_nums, 
            fixed_pool_size=fixed_pool_size,
            depths=depths[::-1], 
            num_stages=num_stages, 
            embed_dims=embed_dims[::-1], 
            window_size=window_size[::-1], 
            mlp_ratios=mlp_ratios[::-1],
            num_heads=num_heads[::-1], 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate, 
            sr_ratios=sr_ratios[::-1],
            t_ratios=t_ratios[::-1], 
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            init_values=init_values, 
            tubelet_size=tubelet_size
        )
        
        self.decoder2D = PretrainVisionTransformerDecoder2D(
            img_size_audio=img_size_audio, 
            patch_size=patch_size, 
            fixed_pool_size=fixed_pool_size,
            depths=depths[::-1], 
            num_stages=num_stages, 
            embed_dims=embed_dims[::-1], 
            window_size=window_size[::-1], 
            mlp_ratios=mlp_ratios[::-1],
            num_heads=num_heads[::-1], 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate, 
            sr_ratios=sr_ratios[::-1],
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            init_values=init_values
        )
        
        self.inter_contrastive_temperature = inter_contrastive_temperature


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

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    def forward(self, x, x_audio, mask=None, mask_audio=None, return_intermediate_features=False):
        
        x_vi, x_au, x_vis_inter_features, x_vis_audio_inter_features = self.encoder(x, x_audio, mask, mask_audio, return_intermediate_features=return_intermediate_features)
        B, N = mask.shape
        _, N_au = mask_audio.shape
        
        _, _, x_vis_inter_features_ori, x_vis_audio_inter_features_ori = self.encoder(x, x_audio, None, None, return_intermediate_features=return_intermediate_features)        
        
        # hcmcl
        logits_per_video, logits_per_audio = [], []
        for x_vis_inter, x_vis_audio_inter, x_vis_inter_ori, x_vis_audio_inter_ori in zip(x_vis_inter_features, x_vis_audio_inter_features, x_vis_inter_features_ori, x_vis_audio_inter_features_ori):
            # pooling
            video_features_inter = x_vis_inter.mean(dim=1)  # (B, C)
            audio_features_inter = x_vis_audio_inter.mean(dim=1)  # (B, C)
            x_vis_inter_ori = x_vis_inter_ori.mean(dim=1).detach()
            x_vis_audio_inter_ori = x_vis_audio_inter_ori.mean(dim=1).detach()

            # normalized features
            video_features_inter = video_features_inter / video_features_inter.norm(dim=1, keepdim=True)
            audio_features_inter = audio_features_inter / audio_features_inter.norm(dim=1, keepdim=True)
            x_vis_inter_ori = x_vis_inter_ori / x_vis_inter_ori.norm(dim=1, keepdim=True)
            x_vis_audio_inter_ori = x_vis_audio_inter_ori / x_vis_audio_inter_ori.norm(dim=1, keepdim=True)

            # cosine similarity as logits
            logits_per_video_inter = video_features_inter @ x_vis_inter_ori.t() / self.inter_contrastive_temperature
            logits_per_audio_inter = audio_features_inter @ x_vis_audio_inter_ori.t() / self.inter_contrastive_temperature

            logits_per_video.append(logits_per_video_inter)
            logits_per_audio.append(logits_per_audio_inter)

        x_vis, _ = self.decoder(x_vi)
        x_audio, _ = self.decoder2D(x_au)
        
        x_vis = x_vis.reshape(B, N, -1)
        x_audio = x_audio.reshape(B, N_au, -1)
        
        _, _, C = x_vis.shape
        _, _, C_au = x_audio.shape
        
        if mask != None:
            x_vis = x_vis[mask] # only return the mask tokens predict pixels
        if mask_audio != None:
            x_audio = x_audio[mask_audio] # only return the mask tokens predict pixels
            
        x_vis = x_vis.reshape(B, -1, C)
        x_audio = x_audio.reshape(B, -1, C_au)
        
        return x_vis, x_audio, logits_per_video, logits_per_audio


@register_model
def pretrain_hicmae_dim512_patch4_160_a256(pretrained=False, **kwargs):
    model = PretrainAudioVisionTransformer(
        img_size=160, 
        patch_size=4, 
        img_size_audio=(256, 128), 
        frame_nums=16, 
        tubelet_size=2, 
        in_chans=3, 
        num_classes=0, 
        embed_dims=[512], 
        window_size=[3], 
        num_heads=[8], 
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
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def pretrain_hicmae_dim512_patch16_160_a256(pretrained=False, **kwargs):
    model = PretrainAudioVisionTransformer(
        img_size=160, 
        patch_size=16, 
        img_size_audio=(256, 128), 
        frame_nums=16, 
        tubelet_size=2, 
        in_chans=3, 
        num_classes=0, 
        embed_dims=[512], 
        window_size=[3], 
        num_heads=[8], 
        mlp_ratios=[4], 
        num_stages=1, 
        depths=[16],
        qkv_bias=False, 
        qk_scale=None, 
        drop_rate=0., 
        attn_drop_rate=0., 
        drop_path_rate=0., 
        norm_layer=nn.LayerNorm, 
        sr_ratios=[4], 
        t_ratios=[2], 
        fixed_pool_size=None, 
        init_values=0.,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def pretrain_mutimae_dim64_512_patch4_160_a256(pretrained=False, **kwargs):
    model = PretrainAudioVisionTransformer(
        img_size=160, 
        patch_size=4,
        img_size_audio=(256, 128), 
        frame_nums=16, 
        tubelet_size=2, 
        in_chans=3, 
        num_classes=0, 
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
        sr_ratios=[1, 4], 
        t_ratios=[1, 2], 
        fixed_pool_size=None, 
        init_values=0.,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

if __name__ == "__main__":
    model = pretrain_hicmae_dim512_patch16_160_a256()
    
    vi = torch.rand((1, 3, 16, 160, 160))
    au = torch.rand((1, 1, 256, 128))
    
    vi_mask = torch.rand((1, 8, 10, 10))>0.5
    au_mask = torch.rand((1, 1, 16, 8))>0.5
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    vi=vi.to(device)
    au=au.to(device)
    vi_mask=vi_mask.to(device).flatten(1)
    au_mask=au_mask.to(device).flatten(1)
    
    re = model(vi, au, vi_mask, au_mask)
    print()
    