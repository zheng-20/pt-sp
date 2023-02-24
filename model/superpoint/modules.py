from typing import List
import torch
import torch.nn as nn

from lib.pointops_sp.functions import pointops_sp
import torch.nn.functional as F

class learn_SLIC_calc_v1_new(nn.Module):
    """
    update features between superpoints and points
    """
    def __init__(self, ch_wc2p_fea: List[int], ch_wc2p_xyz: List[int], ch_mlp: List[int],
                 bn=True, use_xyz=True, use_softmax=True, use_norm=True, last=False):
        super().__init__()
        self.bn = bn
        self.use_xyz = use_xyz
        self.use_softmax = use_softmax
        self.use_norm = use_norm
        self.last = last

        # self.w_c2p_fea = pt_util.SharedMLP(ch_wc2p_fea, bn=self.bn)
        # self.w_c2p_xyz = pt_util.SharedMLP(ch_wc2p_xyz, bn=self.bn)
        # self.mlp = pt_util.SharedMLP_1d(ch_mlp, bn=self.bn)
        self.w_c2p_fea = nn.Sequential(nn.Conv1d(ch_wc2p_fea[0], ch_wc2p_fea[1], kernel_size=1), nn.BatchNorm1d(ch_wc2p_fea[1]), nn.ReLU(inplace=True), nn.Conv1d(ch_wc2p_fea[1], ch_wc2p_fea[2], kernel_size=1))
        self.w_c2p_xyz = nn.Sequential(nn.Conv1d(ch_wc2p_xyz[0], ch_wc2p_xyz[1], kernel_size=1), nn.BatchNorm1d(ch_wc2p_xyz[1]), nn.ReLU(inplace=True), nn.Conv1d(ch_wc2p_xyz[1], ch_wc2p_xyz[2], kernel_size=1))
        self.mlp = nn.Sequential(nn.Linear(ch_mlp[0], ch_mlp[1]), nn.BatchNorm1d(ch_mlp[1]), nn.ReLU(inplace=True), nn.Linear(ch_mlp[1], ch_mlp[2]))

    
    def forward(self, sp_fea, sp_xyz, o_p_fea, p_xyz, c2p_idx_abs, c2p_idx, cluster_idx, offset):
        # sp_fea: b x m x c
        # sp_xyz: b x m x 3
        # o_p_fea: n x c
        # p_xyz: n x 3
        # c2p_idx_abs: n x nc2p 
        # offset: b
        bs = offset.size(0)
        _, n, nc2p = c2p_idx_abs.size()

        c2p_fea = pointops_sp.grouping_offset(sp_fea.transpose(1, 2).contiguous(), c2p_idx_abs, offset).squeeze(0) - o_p_fea.transpose(0, 1).contiguous().unsqueeze(-1).repeat(1, 1, nc2p)
        # c2p_fea: c x n x nc2p 距离每点最近的6个超点特征
        
        c2p_xyz = pointops_sp.grouping_offset(sp_xyz.transpose(1, 2).contiguous(), c2p_idx_abs, offset).squeeze(0) - p_xyz.transpose(0, 1).contiguous().unsqueeze(-1).repeat(1, 1, nc2p)
        # c2p_xyz: 3 x n x nc2p 距离每点最近的6个超点坐标

        p_fea = self.mlp(o_p_fea)    # n x 16
        c2p_fea = self.w_c2p_fea(c2p_fea.transpose(0, 1).contiguous())   # n x 16 x nc2p
        c2p_xyz = self.w_c2p_xyz(c2p_xyz.transpose(0, 1).contiguous())   # n x 16 x nc2p

        # p_fea = p_fea.transpose(1, 2).contiguous().view(bs*n, p_fea.size(1))    # bn x 16
        if self.use_norm:
            p_fea = F.normalize(p_fea, p=2, dim=1)
        
        # w_fea = c2p_fea.transpose(1, 2).contiguous().view(bs*n, c2p_fea.size(1), nc2p)  # bn x 16 x nc2p
        w_fea = c2p_fea
        if self.use_norm:
            w_fea = F.normalize(w_fea, p=2, dim=1)
        
        # w_xyz = c2p_xyz.transpose(1, 2).contiguous().view(bs*n, c2p_xyz.size(1), nc2p)  # bn x 16 x nc2p
        w_xyz = c2p_xyz
        if self.use_norm:
            w_xyz = F.normalize(w_xyz, p=2, dim=1)

        new_w_fea = torch.matmul(p_fea.unsqueeze(1), w_fea)     # (bn, 1, 16) X (bn, 16, nc2p) -> (bn, 1, nc2p)
        new_w_xyz = torch.matmul(p_fea.unsqueeze(1), w_xyz)     # (bn, 1, 16) X (bn, 16, nc2p) -> (bn, 1, nc2p)

        bi_w = (new_w_fea * new_w_xyz).view(n, 6)   # n x nc2p
        if self.use_softmax:
            bi_w = F.softmax(bi_w, dim=-1)  # n x nc2p

        f, sp_nei_cnt = pointops_sp.assomatrixfloat_offset(nc2p, bi_w, c2p_idx, cluster_idx.unsqueeze(-1), offset)
        # f: b x m x n 点与超点中心关联矩阵
        # sp_nei_cnt: b x m x 1 每个超点所包含点数

        # sp_sum = f.sum(dim=2, keepdim=True)                 # b x m x 1
        # sp_fea = torch.matmul(f, o_p_fea) / (sp_sum+1e-8)   # (b, m, n) X (b, n, c) -> (b, m, c)
        
        # sp_xyz = torch.matmul(f, p_xyz) / (sp_sum+1e-8)     # (b, m, n) X (b, n, 3) -> (b, m, 3)
        for i in range(offset.size(0)):
            if i == 0:
                f_i = f[:, :, :offset[0]]
                sp_sum_i = f_i.sum(dim=2, keepdim=True)
                o_p_fea_i = o_p_fea[:offset[0]]
                p_xyz_i = p_xyz[:offset[0]]
                sp_fea = torch.matmul(f_i, o_p_fea_i) / (sp_sum_i+1e-8)
                sp_xyz = torch.matmul(f_i, p_xyz_i) / (sp_sum_i+1e-8)
            else:
                f_i = f[:, :, offset[i-1]:offset[i]]
                sp_sum_i = f_i.sum(dim=2, keepdim=True)
                o_p_fea_i = o_p_fea[offset[i-1]:offset[i]]
                p_xyz_i = p_xyz[offset[i-1]:offset[i]]
                sp_fea_i = torch.matmul(f_i, o_p_fea_i) / (sp_sum_i+1e-8)
                sp_xyz_i = torch.matmul(f_i, p_xyz_i) / (sp_sum_i+1e-8)
                sp_fea = torch.cat([sp_fea, sp_fea_i], 0)   # (b, m, c)
                sp_xyz = torch.cat([sp_xyz, sp_xyz_i], 0)   # (b, m, 3)

        if self.last:
            return bi_w, sp_fea, sp_xyz
        return sp_fea, sp_xyz


def init_fea(p_fea, asso_matrix, sp_nei_cnt, offset):
    # p_fea: b x n x 32
    # asso_matrix: b x m x n
    # sp_nei_cnt: b x m x 1

    for i in range(sp_nei_cnt.size(0)):
        sp_nei_cnt_i = sp_nei_cnt[i].unsqueeze(0)
        if i == 0:
            asso_matrix_i = asso_matrix[:, :, :offset[0]]
            p_fea_i = p_fea[:offset[0]]
            sp_fea = torch.matmul(asso_matrix_i, p_fea_i) / (sp_nei_cnt_i+1e-8)
        else:
            asso_matrix_i = asso_matrix[:, :, offset[i-1]:offset[i]]
            p_fea_i = p_fea[offset[i-1]:offset[i]]
            sp_fea_i = torch.matmul(asso_matrix_i, p_fea_i) / (sp_nei_cnt_i+1e-8)
            sp_fea = torch.cat([sp_fea, sp_fea_i], 0)
        # if torch.isinf(sp_fea).any():
        #     print('wrong!')

    return sp_fea       # b x m x 32


def calc_sp_fea(pt_asso, p_fea, nc2p, c2p_idx, cluster_idx, offset):
    # pt_center_index: b x n x 6
    # pt_asso: b x n x 6 
    # p_fea: b x n x c
    # num: m
    
    # b x m x n
    f, sp_nei_cnt = pointops_sp.assomatrixfloat_offset(nc2p, pt_asso, c2p_idx, cluster_idx.unsqueeze(-1), offset)

    # sp_sum = f.sum(dim=2, keepdim=True)                 # b x m x 1
    # sp_fea = torch.matmul(f, p_fea) / (sp_sum+1e-8)     # (b x m x n) X (b x n x c) -> (b x m x c) 
    for i in range(offset.size(0)):
        if i == 0:
            f_i = f[:, :, :offset[0]]
            sp_sum_i = f_i.sum(dim=2, keepdim=True)
            p_fea_i = p_fea[:, :offset[0]]
            sp_fea = torch.matmul(f_i, p_fea_i) / (sp_sum_i+1e-8)
        else:
            f_i = f[:, :, offset[i-1]:offset[i]]
            sp_sum_i = f_i.sum(dim=2, keepdim=True)
            p_fea_i = p_fea[:, offset[i-1]:offset[i]]
            sp_fea_i = torch.matmul(f_i, p_fea_i) / (sp_sum_i+1e-8)
            sp_fea = torch.cat([sp_fea, sp_fea_i], 0)   # (b, m, c)

    return sp_fea   # b x m x c