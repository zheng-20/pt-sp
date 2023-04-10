from typing import List
import torch
import torch.nn as nn

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
        if self.use_softmax:
            bi_w = F.softmax(bi_w, dim=-1)  # n x nc2p

        # f, sp_nei_cnt = pointops_sp.assomatrixfloat_offset(nc2p, bi_w, c2p_idx, cluster_idx.unsqueeze(-1), offset)
        f, sp_nei_cnt = pointops_sp_v2.assomatrixfloat(nc2p, offset, sp_offset, bi_w, c2p_idx, cluster_idx.unsqueeze(-1))
        # f: b x m x n 点与超点中心关联矩阵
        # sp_nei_cnt: b x m x 1 每个超点所包含点数

        sp_sum = f.sum(dim=1, keepdim=True)                 # b x m x 1
        sp_fea = torch.matmul(f, o_p_fea) / (sp_sum+1e-8)   # (b, m, n) X (b, n, c) -> (b, m, c)
        
        sp_xyz = torch.matmul(f, p_xyz) / (sp_sum+1e-8)     # (b, m, n) X (b, n, 3) -> (b, m, 3)
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