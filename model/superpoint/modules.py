from typing import List
import torch
import torch.nn as nn
from scipy.spatial import KDTree

from lib.pointops_sp.functions import pointops_sp
from lib.pointops_sp_v2.functions import pointops_sp_v2
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


class learn_SLIC_calc_v2(nn.Module):
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

    
    def forward(self, sp_fea, sp_xyz, o_p_fea, p_xyz, c2p_idx_abs, c2p_idx, cluster_idx, offset, sp_offset):
        # sp_fea: b x m x c
        # sp_xyz: b x m x 3
        # o_p_fea: n x c
        # p_xyz: n x 3
        # c2p_idx_abs: n x nc2p 
        # offset: b
        # bs = offset.size(0)
        n, nc2p = c2p_idx_abs.size()

        # c2p_fea = pointops_sp.grouping_offset(sp_fea.transpose(1, 2).contiguous(), c2p_idx_abs, offset).squeeze(0) - o_p_fea.transpose(0, 1).contiguous().unsqueeze(-1).repeat(1, 1, nc2p)
        c2p_fea = pointops_sp_v2.grouping(sp_fea.transpose(0, 1).contiguous(), c2p_idx_abs) - o_p_fea.transpose(0, 1).contiguous().unsqueeze(-1).repeat(1, 1, nc2p)
        # c2p_fea: c x n x nc2p 距离每点最近的6个超点特征
        
        # c2p_xyz = pointops_sp.grouping_offset(sp_xyz.transpose(1, 2).contiguous(), c2p_idx_abs, offset).squeeze(0) - p_xyz.transpose(0, 1).contiguous().unsqueeze(-1).repeat(1, 1, nc2p)
        c2p_xyz = pointops_sp_v2.grouping(sp_xyz.transpose(0, 1).contiguous(), c2p_idx_abs) - p_xyz.transpose(0, 1).contiguous().unsqueeze(-1).repeat(1, 1, nc2p)
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
        # bi_w = (new_w_fea).view(n, 6)   # n x nc2p
        if self.use_softmax:
            bi_w = F.softmax(bi_w, dim=-1)  # n x nc2p

        # f, sp_nei_cnt = pointops_sp.assomatrixfloat_offset(nc2p, bi_w, c2p_idx, cluster_idx.unsqueeze(-1), offset)
        f, sp_nei_cnt = pointops_sp_v2.assomatrixfloat(nc2p, offset, sp_offset, bi_w, c2p_idx, cluster_idx.unsqueeze(-1))
        # f: b x m x n 点与超点中心关联概率矩阵
        # sp_nei_cnt: b x m x 1 每个超点所包含点数

        sp_sum = f.sum(dim=1, keepdim=True)                 # b x m x 1
        sp_fea = torch.matmul(f, o_p_fea) / (sp_sum+1e-8)   # (b, m, n) X (b, n, c) -> (b, m, c)
        
        sp_xyz = torch.matmul(f, p_xyz) / (sp_sum+1e-8)     # (b, m, n) X (b, n, 3) -> (b, m, 3)
        idx = []
        # # 查找超点中的最近邻点
        # p2sp_idx = torch.argmax(bi_w, dim=1)                   # n 点-超点硬关联
        # sp_idx = torch.gather(c2p_idx_abs, 1, p2sp_idx.view(-1, 1)).squeeze()     # n
        # if self.last:
        #     # idx = []
        #     for j in range(sp_xyz.size(0)):
        #         # try:
        #         p_xyz_j = p_xyz[sp_idx == j]
        #         p_xyz_j_idx = torch.nonzero(sp_idx == j).squeeze()
        #         if (p_xyz_j_idx.numel() == 1):  # 超点中只有一个点，选择该点作为超点坐标
        #             # idx.append(p_xyz_j_idx)
        #             if j == 0:
        #                 idx = p_xyz_j_idx.unsqueeze(0)
        #             else:
        #                 idx = torch.cat((idx, p_xyz_j_idx.unsqueeze(0)))
        #             continue
        #         if (p_xyz_j_idx.numel() == 0):  # 超点中没有点，选择去全局最近点作为超点坐标
        #             # tree = KDTree(p_xyz.cpu().numpy())  # 构建KDTree
        #             # dist, idx_j = tree.query(sp_xyz[j].detach().cpu().numpy(), k=1)  # 每个超点最近的点
        #             # idx.append(idx_j)
        #             dist = torch.cdist(sp_xyz[j].unsqueeze(0), p_xyz)
        #             index = torch.argmin(dist).unsqueeze(0)
        #             if j == 0:
        #                 idx = index
        #             else:
        #                 idx = torch.cat((idx, index))
        #             continue
        #         # tree = KDTree(p_xyz_j.cpu().numpy())  # 构建KDTree
        #         # dist, idx_j = tree.query(sp_xyz[j].detach().cpu().numpy(), k=1)  # 分配到每个超点的点中距离超点中心最近的点
        #         # idx.append(p_xyz_j_idx[idx_j])
        #         dist = torch.cdist(sp_xyz[j].unsqueeze(0), p_xyz_j)
        #         index = p_xyz_j_idx[torch.argmin(dist)].unsqueeze(0)
        #         if j == 0:
        #             idx = index
        #         else:
        #             idx = torch.cat((idx, index))
        #         # except:
        #         #     print('error')
        #     # idx = torch.tensor(idx).cuda()
        #     sp_xyz = p_xyz[idx, :]  # 选择最近点作为超点坐标
        #     # sp_fea = o_p_fea[idx, :]  # 选择最近点作为超点特征

        # # 查找全局点云中的最近邻点
        # tree = KDTree(p_xyz.cpu().numpy())  # 构建KDTree
        # dist, idx = tree.query(sp_xyz.detach().cpu().numpy(), k=1)  # 每个超点最近的点
        # idx = torch.tensor(idx).cuda()
        # nearest_coord = p_xyz[idx, :]  # m x 3
        # sp_xyz = nearest_coord  # 选择最近点作为超点坐标


        # for i in range(offset.size(0)):
        #     if i == 0:
        #         f_i = f[:, :, :offset[0]]
        #         sp_sum_i = f_i.sum(dim=2, keepdim=True)
        #         o_p_fea_i = o_p_fea[:offset[0]]
        #         p_xyz_i = p_xyz[:offset[0]]
        #         sp_fea = torch.matmul(f_i, o_p_fea_i) / (sp_sum_i+1e-8)
        #         sp_xyz = torch.matmul(f_i, p_xyz_i) / (sp_sum_i+1e-8)
        #     else:
        #         f_i = f[:, :, offset[i-1]:offset[i]]
        #         sp_sum_i = f_i.sum(dim=2, keepdim=True)
        #         o_p_fea_i = o_p_fea[offset[i-1]:offset[i]]
        #         p_xyz_i = p_xyz[offset[i-1]:offset[i]]
        #         sp_fea_i = torch.matmul(f_i, o_p_fea_i) / (sp_sum_i+1e-8)
        #         sp_xyz_i = torch.matmul(f_i, p_xyz_i) / (sp_sum_i+1e-8)
        #         sp_fea = torch.cat([sp_fea, sp_fea_i], 0)   # (b, m, c)
        #         sp_xyz = torch.cat([sp_xyz, sp_xyz_i], 0)   # (b, m, 3)

        if self.last:
            return bi_w, sp_fea, sp_xyz, idx
        return sp_fea, sp_xyz


class superpoint_transformer(nn.Module):
    """
    利用transformer编码点与超点中心之间的特征关系, 构建返回association map
    """
    def __init__(self, ch_wc2p_fea: List[int], ch_wc2p_xyz: List[int], ch_mlp: List[int],
                 bn=True, use_xyz=True, use_softmax=True, use_norm=True, last=False):
        super(superpoint_transformer, self).__init__()
        self.bn = bn
        self.use_xyz = use_xyz
        self.use_softmax = use_softmax
        self.use_norm = use_norm
        self.last = last
        self.share_planes = 2
        in_planes = ch_wc2p_fea[0]
        mid_planes = ch_wc2p_fea[0]
        out_planes = ch_wc2p_fea[1]

        self.linear_q = nn.Linear(in_planes, mid_planes)
        self.linear_k = nn.Linear(in_planes, mid_planes)
        self.linear_v = nn.Linear(in_planes, mid_planes)
        self.linear_p = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3), nn.ReLU(inplace=True), nn.Linear(3, mid_planes))
        self.linear_w = nn.Sequential(nn.BatchNorm1d(mid_planes), nn.ReLU(inplace=True),
                                    nn.Linear(mid_planes, out_planes),
                                    nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True),
                                    nn.Linear(out_planes, out_planes))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, sp_fea, sp_xyz, o_p_fea, p_xyz, c2p_idx_abs, c2p_idx, cluster_idx, offset, sp_offset):
        # sp_fea: b x m x c
        # sp_xyz: b x m x 3
        # o_p_fea: n x c
        # p_xyz: n x 3
        # c2p_idx_abs: n x nc2p 
        # offset: b
        
        n, nc2p = c2p_idx_abs.size()    # n: 点云中点的个数, nc2p: 每个点最近的超点个数
        q_p_fea = self.linear_q(o_p_fea)   # n x c
        k_sp_fea = self.linear_k(sp_fea)   # m x c
        v_sp_fea = self.linear_v(sp_fea)   # m x c_out
        k_sp_fea = pointops_sp_v2.grouping(k_sp_fea.transpose(0, 1).contiguous(), c2p_idx_abs).permute(1, 2, 0).contiguous()  # n x nc2p x c
        v_sp_fea = pointops_sp_v2.grouping(v_sp_fea.transpose(0, 1).contiguous(), c2p_idx_abs).permute(1, 2, 0).contiguous()  # n x nc2p x c
        p_e = pointops_sp_v2.grouping(sp_xyz.transpose(0, 1).contiguous(), c2p_idx_abs).permute(1, 2, 0).contiguous()  # n x nc2p x 3
        for i, layer in enumerate(self.linear_p): p_e = layer(p_e.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i == 1 else layer(p_e)    # (n, nc2p, c)
        w = k_sp_fea - q_p_fea.unsqueeze(1).repeat(1, nc2p, 1) + p_e    # (n, nc2p, c)
        for i, layer in enumerate(self.linear_w): w = layer(w.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i % 3 == 0 else layer(w)
        w = self.softmax(w)  # (n, nsample, c)
        c = v_sp_fea.shape[2]; s = self.share_planes
        bi_w = ((v_sp_fea + p_e).view(n, nc2p, s, c // s) * w.unsqueeze(2)).view(n, nc2p, c)  # (n, nc2p, c)
        bi_w = bi_w.sum(2).view(n, nc2p)  # (n, nc2p)

        asso_matrix = F.softmax(bi_w, dim=1)  # (n, nc2p)

        return asso_matrix


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


def calc_sp_fea_v2(pt_asso, p_fea, nc2p, c2p_idx, cluster_idx, offset, sp_offset):
    # pt_center_index: b x n x 6
    # pt_asso: b x n x 6 
    # p_fea: b x n x c
    # num: m
    
    # b x m x n
    f, sp_nei_cnt = pointops_sp_v2.assomatrixfloat(nc2p, offset, sp_offset, pt_asso, c2p_idx, cluster_idx.unsqueeze(-1))

    sp_sum = f.sum(dim=1, keepdim=True)                 # b x m x 1
    sp_fea = torch.matmul(f, p_fea) / (sp_sum+1e-8)     # (b x m x n) X (b x n x c) -> (b x m x c) 
    # for i in range(offset.size(0)):
    #     if i == 0:
    #         f_i = f[:, :, :offset[0]]
    #         sp_sum_i = f_i.sum(dim=2, keepdim=True)
    #         p_fea_i = p_fea[:, :offset[0]]
    #         sp_fea = torch.matmul(f_i, p_fea_i) / (sp_sum_i+1e-8)
    #     else:
    #         f_i = f[:, :, offset[i-1]:offset[i]]
    #         sp_sum_i = f_i.sum(dim=2, keepdim=True)
    #         p_fea_i = p_fea[:, offset[i-1]:offset[i]]
    #         sp_fea_i = torch.matmul(f_i, p_fea_i) / (sp_sum_i+1e-8)
    #         sp_fea = torch.cat([sp_fea, sp_fea_i], 0)   # (b, m, c)

    return sp_fea   # b x m x c


def calc_sp_normal(pt_asso, p_normal, nc2p, c2p_idx, cluster_idx, offset):
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
            p_normal_i = p_normal[:, :offset[0]]
            sp_normal = torch.matmul(f_i, p_normal_i) / (sp_sum_i+1e-8)
        else:
            f_i = f[:, :, offset[i-1]:offset[i]]
            sp_sum_i = f_i.sum(dim=2, keepdim=True)
            p_normal_i = p_normal[:, offset[i-1]:offset[i]]
            sp_normal_i = torch.matmul(f_i, p_normal_i) / (sp_sum_i+1e-8)
            sp_normal = torch.cat([sp_normal, sp_normal_i], 0)   # (b, m, c)

    return sp_normal   # b x m x c


def point_normal_similarity_loss(normals, p2sp_idx, c2p_idx_abs, offset):
    """
    计算点云法向量相似度最小化的损失函数
    :param normals: tensor, [num_points, 3], 点云法向量
    :param neighbors_idx: tensor, [batch_size, num_points, k], 每个点的k近邻的索引
    :p2sp_idx: tensor, [num_points], 每个点分配到的超点索引, 基于nc2p == 6
    :c2p_idx_abs: tensor, [1, num_points, nc2p], 每个点最近的nc2p个超点中心索引, 基于m
    :offset: tensor, [batch_size], 点云对象起始位置标识
    :param k: int, 每个点的邻居数量
    :return: tensor, [1], 损失函数值
    """

    sp_idx = torch.gather(c2p_idx_abs.squeeze(0), dim=1, index=p2sp_idx.view(-1,1)).squeeze(1) # [num_points]
    unique_idx = torch.unique(sp_idx)
    loss = torch.tensor(0.0, device=normals.device)

    for i in range(len(offset)):
        if i == 0:
            normal_i = normals[:offset[0]]
            sp_idx_i = sp_idx[:offset[0]]
        else:
            normal_i = normals[offset[i-1]:offset[i]]
            sp_idx_i = sp_idx[offset[i-1]:offset[i]]
        for j in unique_idx:
            norm = normal_i[sp_idx_i == j]
            num_points = len(norm)
            if num_points == 0:
                # print('l')
                continue
            # dot_matrix = torch.matmul(norm, norm.T)
            # norms = torch.norm(norm, dim=1)
            # similarity_matrix = dot_matrix / torch.matmul(norms.view(-1,1), norms.view(1,-1))
            # loss += (1 - similarity_matrix).sum() / (num_points * num_points)
            norm_vectors = F.normalize(norm, dim=1)
            similarity_matrix = torch.matmul(norm_vectors, norm_vectors.t())
            similarity_matrix.fill_diagonal_(0)
            similarity_matrix = F.relu(similarity_matrix)
            loss += similarity_matrix.mean()
            # loss += 1 - similarity_matrix.mean()
    
    loss.requires_grad = True

    # batch_size, num_points, _ = normals.size()
    # device = normals.device
    
    # # 获取每个点的法向量和邻居点的法向量
    # norm_i = normals.unsqueeze(dim=2).repeat(1, 1, k, 1)  # [batch_size, num_points, k, 3]
    # norm_j = torch.gather(normals.unsqueeze(dim=1).repeat(1, num_points, 1, 1),
    #                       index=neighbors_idx.unsqueeze(dim=-1).repeat(1, 1, 1, 3), dim=1)  # [batch_size, num_points, k, 3]

    # # 计算每个点的法向量与邻居点法向量的余弦相似度
    # similarity = torch.cosine_similarity(norm_i, norm_j, dim=-1)  # [batch_size, num_points, k]
    
    # # 计算每个点的损失函数值
    # loss = (1 - similarity).sum() / (batch_size * num_points * k)
    
    return loss / len(offset)


def Contrastive_InfoNCE_loss_sp(sp_fea, sp_xyz_idx, instance_label, sp_offset, temperature=1.0):
    """
    计算超点中心的对比损失函数
    :param sp_fea: tensor, [m, 32], 超点的特征
    :param sp_xyz_idx: tensor, [m], 超点的索引(基于n)
    :param instance_label: tensor, [n], 每个点的基元实例标签
    :param sp_offset: tensor, [batch_size], 超点起始位置标识
    :param temperature: float, 温度参数
    """

    loss = torch.tensor(0.0, device=sp_fea.device)
    sp_instance_label = instance_label[sp_xyz_idx] # [m]
    for i in range(len(sp_offset)):
        if i == 0:
            sp_fea_i = sp_fea[:sp_offset[0]]
            sp_instance_label_i = sp_instance_label[:sp_offset[0]]
        else:
            sp_fea_i = sp_fea[sp_offset[i-1]:sp_offset[i]]
            sp_instance_label_i = sp_instance_label[sp_offset[i-1]:sp_offset[i]]
        # 计算每个点云模型中超点的对比损失函数
        m = sp_fea_i.size(0)
        sp_neighbor_fea = sp_fea_i.unsqueeze(0).repeat(m, 1, 1) # [m, m, 32]   
        mask = torch.ones(m, m)
        idx = torch.arange(m)
        mask = mask.masked_fill(idx == idx[:, None], 0).to(device=sp_fea.device) # [m, m]
        sp_neighbor_fea = sp_neighbor_fea[mask.bool()].view(m, -1, 32)  # [m, m-1, 32]

        sp_neighbor_instance_label = sp_instance_label_i.unsqueeze(0).repeat(m, 1) # [m, m]
        sp_neighbor_instance_label = sp_neighbor_instance_label[mask.bool()].view(m, -1)  # [m, m-1]

        posmask = sp_neighbor_instance_label == sp_instance_label_i.unsqueeze(1)
        point_mask = torch.sum(posmask.int(), -1)
        point_mask = torch.logical_and(0 < point_mask, point_mask < m - 1)

        if not torch.any(point_mask):
            continue

        posmask = posmask[point_mask]
        sp_fea_i = sp_fea_i[point_mask]
        sp_neighbor_fea = sp_neighbor_fea[point_mask]

        dist_l2 = torch.sum((sp_fea_i.unsqueeze(1) - sp_neighbor_fea) ** 2, dim=-1)
        dist_l2 = torch.sqrt(dist_l2 + 1e-8)

        dist_l2 = -dist_l2
        dist_l2 = dist_l2 - torch.max(dist_l2, -1, keepdim=True)[0]
        exp = torch.exp(dist_l2 / temperature)

        pos = torch.sum(exp * posmask, dim=-1)
        neg = torch.sum(exp, dim=-1)
        infoNCE_loss = -torch.log(pos / neg + 1e-8).mean()
        loss += infoNCE_loss
    
    return loss / len(sp_offset)


def Contrastive_InfoNCE_loss_p2sp(p_fea, p2sp_idx, c2p_idx_abs, sp_fea, sp_xyz_idx, instance_label, sp_offset, temperature=1):
    """
    计算点与超点的对比损失函数
    :param p_fea: tensor, [n, 32], 点的特征
    :param p2sp_idx: tensor, [n], 点对应的超点索引(基于k)
    :param c2p_idx_abs: tensor, [n], 点对应的最近6个超点索引(基于m)
    :param sp_fea: tensor, [m, 32], 超点的特征
    :param sp_xyz_idx: tensor, [m], 超点的索引(基于n)
    :param instance_label: tensor, [n], 每个点的基元实例标签
    :param sp_offset: tensor, [batch_size], 超点起始位置标识
    :param temperature: float, 温度参数
    """

    loss = torch.tensor(0.0, device=sp_fea.device)
    sp_instance_label = instance_label[sp_xyz_idx] # [m]超点中心对应的实例标签
    sp_idx = torch.gather(c2p_idx_abs.squeeze(0), dim=1, index=p2sp_idx.view(-1,1)).squeeze(1) # [n]点对应的超点索引(基于m)
    for i in range(len(sp_offset)):
        if i == 0:
            sp_fea_i = sp_fea[:sp_offset[0]]
            sp_instance_label_i = sp_instance_label[:sp_offset[0]]
        else:
            sp_fea_i = sp_fea[sp_offset[i-1]:sp_offset[i]]
            sp_instance_label_i = sp_instance_label[sp_offset[i-1]:sp_offset[i]]
        # 计算每个点云模型中每个超点中相同实例label的点与超点中心的对比损失
        m = sp_fea_i.size(0)    # 当前点云超点中心数量
        for j in range(m):
            p_fea_j = p_fea[sp_idx == j]    # 分配到当前超点的点特征
            p_label_j = instance_label[sp_idx == j] # 分配到当前超点的点实例label
            sp_fea_j = sp_fea_i[j]  # 当前超点特征
            sp_instance_label_j = sp_instance_label_i[j]    # 当前超点实例label
            k = p_fea_j.size(0) # 分配到当前超点的点数量
            posmask = p_label_j == sp_instance_label_j
            point_mask = torch.sum(posmask.int())
            point_mask = torch.logical_and(0 < point_mask, point_mask < k - 1)

            if not point_mask:
                continue

            dist_l2 = torch.sum((sp_fea_j.unsqueeze(0) - p_fea_j) ** 2, dim=-1)
            dist_l2 = torch.sqrt(dist_l2 + 1e-8)

            dist_l2 = -dist_l2
            dist_l2 = dist_l2 - torch.max(dist_l2, -1, keepdim=True)[0]
            exp = torch.exp(dist_l2 / temperature)

            pos = torch.sum(exp * posmask, dim=-1)
            neg = torch.sum(exp, dim=-1)
            infoNCE_loss = -torch.log(pos / neg + 1e-8).mean()
            loss += infoNCE_loss
    
    return loss / sp_offset[-1]


def Contrastive_InfoNCE_loss_re_p_fea(re_p_fea, p2sp_idx, c2p_idx_abs, instance_label, offset, temperature=1.0):
    """
    计算重建点特征的对比损失函数
    :param re_p_fea: tensor, [n, 32], 重建的逐点特征
    :param p2sp_idx: tensor, [n], 点对应的超点索引(基于k)
    :param c2p_idx_abs: tensor, [n], 点对应的最近6个超点索引(基于m)
    :param instance_label: tensor, [n], 每个点的基元实例标签
    :param offset: tensor, [batch_size], 点起始位置标识
    :param temperature: float, 温度参数
    """

    loss = torch.tensor(0.0, device=re_p_fea.device)
    sp_idx = torch.gather(c2p_idx_abs.squeeze(0), dim=1, index=p2sp_idx.view(-1,1)).squeeze(1) # [n]点对应的超点索引(基于m)
    m = torch.max(sp_idx) + 1
    # 计算每个超点中相同实例标签的重建点特征的对比损失函数
    for i in range(m):
        fea_i = re_p_fea[sp_idx == i]   # 分配到当前超点的重建点特征
        label_i = instance_label[sp_idx == i] # 分配到当前超点的重建点实例label
        k = fea_i.size(0) # 分配到当前超点的点数量
        neighbor_fea_i = fea_i.unsqueeze(0).repeat(k, 1, 1)
        neighbor_label_i = label_i.unsqueeze(0).repeat(k, 1)
        posmask = neighbor_label_i == label_i.unsqueeze(1)
        point_mask = torch.sum(posmask.int(), -1)
        point_mask = torch.logical_and(0 < point_mask, point_mask < k - 1)

        if not torch.any(point_mask):
            continue

        # posmask = posmask[point_mask]
        # fea_i = fea_i[point_mask]
        # neighbor_fea_i = neighbor_fea_i[point_mask]

        dist_l2 = torch.sum((fea_i.unsqueeze(1) - neighbor_fea_i) ** 2, dim=-1)
        dist_l2 = torch.sqrt(dist_l2 + 1e-8)

        dist_l2 = -dist_l2
        dist_l2 = dist_l2 - torch.max(dist_l2, -1, keepdim=True)[0]
        exp = torch.exp(dist_l2 / temperature)

        pos = torch.sum(exp * posmask, dim=-1)
        neg = torch.sum(exp, dim=-1)
        infoNCE_loss = -torch.log(pos / neg + 1e-8).mean()
        loss += infoNCE_loss

    return loss / m


def infoNCE_loss_p2sp(sp_fea, p_fea, c2p_idx_abs, c2p_idx, instance_label, temperature=1.0):
    '''
    每次迭代后计算点与超点特征的对比损失函数
    :param sp_fea: tensor, [m, 32], 超点特征
    :param p_fea: tensor, [n, 32], 点特征
    :param c2p_idx_abs: tensor, [n, 6], 点对应的最近6个超点索引(基于m)
    :param c2p_idx: tensor, [n, 6], 点对应的最近6个超点索引(基于n)
    :param instance_label: tensor, [n], 每个点的基元实例标签
    :param temperature: float, 温度参数
    '''
    c2p_fea = pointops_sp_v2.grouping(sp_fea.transpose(0, 1).contiguous(), c2p_idx_abs)  # c x n x 6 每点最近的k个超点特征
    init_sp_instance = instance_label[c2p_idx.long()]   # n x k 初始化每个点最近的k个超点实例label
    posmask = init_sp_instance == instance_label.unsqueeze(-1)
    point_mask = torch.sum(posmask.int(), dim=1)
    point_mask = torch.logical_and(point_mask > 0, point_mask < 6)  # 保证每个点最近的6个超点都有正负样本

    if not torch.any(point_mask):
        return torch.tensor(0.0, device=sp_fea.device)

    posmask = posmask[point_mask]   # 正样本掩码
    contrast_p_fea = p_fea[point_mask]  # 点特征
    contrast_sp_fea = c2p_fea.permute(1, 2, 0).contiguous()[point_mask] # 点最近的6个超点特征

    dist_l2 = torch.sum((contrast_p_fea.unsqueeze(1) - contrast_sp_fea) ** 2, dim=-1)
    dist_l2 = torch.sqrt(dist_l2 + 1e-8)
    dist_l2 = -dist_l2
    dist_l2 = dist_l2 - torch.max(dist_l2, -1, keepdim=True)[0]
    exp = torch.exp(dist_l2 / temperature)
    pos = torch.sum(exp * posmask, dim=-1)
    neg = torch.sum(exp, dim=-1)
    loss = -torch.log(pos / neg + 1e-8).mean()

    return loss