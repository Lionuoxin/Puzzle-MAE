import torch.nn as nn
import torch
import math
import numpy as np

class CoPE(nn.Module):
    def __init__(self, npos_max, head_dim, H=None, W=None, T=None):
        super().__init__()
        self.npos_max = npos_max
        self.pos_emb = nn.parameter.Parameter(
        torch.zeros(1, head_dim, npos_max))
        self.H = H
        self.W = W
        self.T = T

    def forward(self, query, attn_logits):
        # compute positions
        gates = torch.sigmoid(attn_logits)
        batch, head_num, q_len, k_len = gates.shape 
        if (self.H == None):
            pos = gates.flip(-1).cumsum(dim=-1).flip(-1)
        else:
            gates = gates.flip(-1).reshape(batch, head_num, q_len, self.H, self.W, self.T)
            pos = gates.cumsum(dim=-1).cumsum(dim=-2).cumsum(dim=-3).reshape(batch, head_num, q_len, k_len).flip(-1)
        pos = pos.clamp(max=self.npos_max- 1)
        # interpolate from integer positions
        pos_ceil = pos.ceil().long()
        pos_floor = pos.floor().long()
        logits_int = torch.matmul(query, self.pos_emb)
        logits_ceil = logits_int.gather(-1, pos_ceil)
        logits_floor = logits_int.gather(-1, pos_floor)
        w = pos- pos_floor
        return logits_ceil * w + logits_floor * (1- w)

class CoPE2D(nn.Module):
    def __init__(self, npos_max, head_dim, H=None, W=None):
        super().__init__()
        self.npos_max = npos_max
        self.pos_emb = nn.parameter.Parameter(
        torch.zeros(1, head_dim, npos_max))
        self.H = H
        self.W = W

    def forward(self, query, attn_logits):
        # compute positions
        gates = torch.sigmoid(attn_logits)
        batch, head_num, q_len, k_len = gates.shape 
        if (self.H == None):
            pos = gates.flip(-1).cumsum(dim=-1).flip(-1)
        else:
            gates = gates.flip(-1).reshape(batch, head_num, q_len, self.H, self.W)
            pos = gates.cumsum(dim=-1).cumsum(dim=-2).reshape(batch, head_num, q_len, k_len).flip(-1)
        pos = pos.clamp(max=self.npos_max- 1)
        # interpolate from integer positions
        pos_ceil = pos.ceil().long()
        pos_floor = pos.floor().long()
        logits_int = torch.matmul(query, self.pos_emb)
        logits_ceil = logits_int.gather(-1, pos_ceil)
        logits_floor = logits_int.gather(-1, pos_floor)
        w = pos- pos_floor
        return logits_ceil * w + logits_floor * (1- w)

class SelfAttn(nn.Module):
    def __init__(self, npos_max, head_dim):
        super().__init__()
        self.cope = CoPE(npos_max, head_dim)
        self.head_dim = head_dim

    def forward(self, query, key, val, mask=None):
        # q, k, v have dimensions batch x seq_len x head_dim
        attn_logits = (query.unsqueeze(-2) @ key.transpose(-1,-2)).squeeze(-2)
        attn_logits = attn_logits / math.sqrt(self.head_dim)
        if(mask != None):
            attn_logits += mask.log()
        attn_logits += self.cope(query, attn_logits)
        attn = torch.softmax(attn_logits, dim=-1)
        out = (attn.unsqueeze(-2) @ val).squeeze(-2)
        return out
    