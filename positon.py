import torch
from timm.models.layers import DropPath, to_2tuple
import torch.nn.functional as F

def get_relative_position_cpb(query_size, key_size, pretrain_size=None):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    pretrain_size = pretrain_size or query_size
    axis_qh = torch.arange(query_size[0], dtype=torch.float32, device=device)
    axis_kh = F.adaptive_avg_pool1d(axis_qh.unsqueeze(0), key_size[0]).squeeze(0)
    axis_qw = torch.arange(query_size[1], dtype=torch.float32, device=device)
    axis_kw = F.adaptive_avg_pool1d(axis_qw.unsqueeze(0), key_size[1]).squeeze(0)
    axis_qt = torch.arange(query_size[2], dtype=torch.float32, device=device)
    axis_kt = F.adaptive_avg_pool1d(axis_qt.unsqueeze(0), key_size[2]).squeeze(0)

    axis_kh, axis_kw, axis_kt = torch.meshgrid(axis_kh, axis_kw, axis_kt)
    axis_qh, axis_qw, axis_qt = torch.meshgrid(axis_qh, axis_qw, axis_qt)

    axis_kh = torch.reshape(axis_kh, [-1])
    axis_kw = torch.reshape(axis_kw, [-1])
    axis_kt = torch.reshape(axis_kt, [-1])
    axis_qh = torch.reshape(axis_qh, [-1])
    axis_qw = torch.reshape(axis_qw, [-1])
    axis_qt = torch.reshape(axis_qt, [-1])

    relative_h = (axis_qh[:, None] - axis_kh[None, :]) / (pretrain_size[0] - 1) * 8
    relative_w = (axis_qw[:, None] - axis_kw[None, :]) / (pretrain_size[1] - 1) * 8
    relative_t = (axis_qt[:, None] - axis_kt[None, :]) / (pretrain_size[2] - 1) * 8
    relative_hwt = torch.stack([relative_h, relative_w, relative_t], dim=-1).view(-1, 3)

    relative_coords_table, idx_map = torch.unique(relative_hwt, return_inverse=True, dim=0)

    relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
        torch.abs(relative_coords_table) + 1.0) / torch.log2(torch.tensor(8, dtype=torch.float32))

    return idx_map, relative_coords_table

def get_relative_position_cpb_(query_size, key_size, pretrain_size=None):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    pretrain_size = pretrain_size or query_size
    axis_qh = torch.arange(query_size[0], dtype=torch.float32, device=device)
    axis_kh = F.adaptive_avg_pool1d(axis_qh.unsqueeze(0), key_size[0]).squeeze(0)
    axis_qw = torch.arange(query_size[1], dtype=torch.float32, device=device)
    axis_kw = F.adaptive_avg_pool1d(axis_qw.unsqueeze(0), key_size[1]).squeeze(0)
    axis_kh, axis_kw = torch.meshgrid(axis_kh, axis_kw)
    axis_qh, axis_qw = torch.meshgrid(axis_qh, axis_qw)

    axis_kh = torch.reshape(axis_kh, [-1])
    axis_kw = torch.reshape(axis_kw, [-1])
    axis_qh = torch.reshape(axis_qh, [-1])
    axis_qw = torch.reshape(axis_qw, [-1])

    relative_h = (axis_qh[:, None] - axis_kh[None, :]) / (pretrain_size[0] - 1) * 8
    relative_w = (axis_qw[:, None] - axis_kw[None, :]) / (pretrain_size[1] - 1) * 8
    relative_hw = torch.stack([relative_h, relative_w], dim=-1).view(-1, 2)

    relative_coords_table, idx_map = torch.unique(relative_hw, return_inverse=True, dim=0)

    relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
        torch.abs(relative_coords_table) + 1.0) / torch.log2(torch.tensor(8, dtype=torch.float32))

    return idx_map, relative_coords_table

def get_relative_position_cpb_2D(query_size, key_size, pretrain_size=None):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    pretrain_size = pretrain_size or query_size
    axis_qh = torch.arange(query_size[0], dtype=torch.float32, device=device)
    axis_kh = F.adaptive_avg_pool1d(axis_qh.unsqueeze(0), key_size[0]).squeeze(0)
    axis_qw = torch.arange(query_size[1], dtype=torch.float32, device=device)
    axis_kw = F.adaptive_avg_pool1d(axis_qw.unsqueeze(0), key_size[1]).squeeze(0)

    axis_kh, axis_kw = torch.meshgrid(axis_kh, axis_kw)
    axis_qh, axis_qw = torch.meshgrid(axis_qh, axis_qw)

    axis_kh = torch.reshape(axis_kh, [-1])
    axis_kw = torch.reshape(axis_kw, [-1])
    axis_qh = torch.reshape(axis_qh, [-1])
    axis_qw = torch.reshape(axis_qw, [-1])

    relative_h = (axis_qh[:, None] - axis_kh[None, :]) / (pretrain_size[0] - 1) * 8
    relative_w = (axis_qw[:, None] - axis_kw[None, :]) / (pretrain_size[1] - 1) * 8
    relative_hw = torch.stack([relative_h, relative_w], dim=-1).view(-1, 2)

    relative_coords_table, idx_map = torch.unique(relative_hw, return_inverse=True, dim=0)

    relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
        torch.abs(relative_coords_table) + 1.0) / torch.log2(torch.tensor(8, dtype=torch.float32))

    return idx_map, relative_coords_table

if __name__ == "__main__":
    print(1)
    i = 0
    img_size = 224
    sr_ratios=[8, 4, 2, 1]
    fixed_pool_size = None
    pretrain_size = None or img_size

    relative_pos_index, relative_coords_table = get_relative_position_cpb(
                    # query_size=(224 // (2 ** (i + 2)), 224 // (2 ** (i + 2)), 15 // (2 ** (i + 1))),
                    # key_size=(224 // ((2 ** (i + 2)) * sr_ratios[i]), 224 // ((2 ** (i + 2)) * sr_ratios[i]), int(15 // ((2 ** (i-1)) * sr_ratios[i]))),
                    # pretrain_size=(224 // (2 ** (i + 2)), 224 // (2 ** (i + 2)), 15 // (2 ** (i + 2)))
                    query_size=(10, 10, 8),
                    key_size=(3, 3, 2),
                    pretrain_size=(10, 10, 8)
    )


    relative_coords_table.transpose(0, 1)[:, relative_pos_index.view(-1)].view(-1, 10*10*8, 3*3*2)
    
    
    relative_pos_index, relative_coords_table = get_relative_position_cpb_(
                    query_size=(56, 56),
                    key_size=(7, 7),
                    pretrain_size=(56, 56)
    )


    relative_coords_table.transpose(0, 1)[:, relative_pos_index.view(-1)].view(-1, 56*56, 7*7)

    print(1)

    # patches num
    # 224 224 16
    # 56  56  16
    # 28  28  8
    # 14  14  4
    # 7   7   2