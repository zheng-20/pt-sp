from typing import Tuple

import torch
from torch.autograd import Function
import torch.nn as nn

import pointops_sp_v2_cuda

class Gathering(Function):
    @staticmethod
    def forward(ctx, features, idx):
        """
        input: features: (c, n), idx : (m) tensor
        output: (c, m)
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()
        c, n = features.size()
        m = idx.size(0)
        output = torch.cuda.FloatTensor(c, m)
        pointops_sp_v2_cuda.gathering_forward_cuda(c, n, m, features, idx, output)
        ctx.for_backwards = (idx, c, n)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        idx, c, n = ctx.for_backwards
        m = idx.size(0)
        grad_features = torch.cuda.FloatTensor(c, n).zero_()
        grad_out_data = grad_out.data.contiguous()
        pointops_sp_v2_cuda.gathering_backward_cuda(c, n, m, grad_out_data, idx, grad_features.data)
        return grad_features, None

gathering = Gathering.apply


################################################################
# ---------   knn query clusters for each point ----------------
################################################################
class KNNQueryCluster(Function):
    @staticmethod
    def forward(ctx, nsample: int, xyz: torch.Tensor, xyz_idx: torch.Tensor, new_xyz: torch.Tensor, offset: torch.Tensor, sp_offset: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        KNN Indexing
        input: nsample: int32, Number of neighbor
               xyz: (n, 3) coordinates of the features
               xyz_idx: (n)  index of the coordinates
               new_xyz: (m, 3) centriods
               offset: (b) offset of each batch
               sp_offset: (b) offset of each superpoint
            output: idx: (m, nsample)
                   ( dist2: (m, nsample) )
        """
        if new_xyz is None:
            new_xyz = xyz
        assert xyz.is_contiguous()
        assert xyz_idx.is_contiguous()
        assert new_xyz.is_contiguous()
        m, _ = new_xyz.size()
        n = xyz.size(0)
        idx = torch.cuda.IntTensor(m, nsample).zero_()
        idx_abs = torch.cuda.IntTensor(m, nsample).zero_()
        dist2 = torch.cuda.FloatTensor(m, nsample).zero_()
        pointops_sp_v2_cuda.knnquerycluster_cuda(n, m, nsample, xyz, xyz_idx, new_xyz, offset, sp_offset, idx, idx_abs, dist2)
        return idx, idx_abs

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None, None, None

knnquerycluster = KNNQueryCluster.apply

################################################################
# ---------   association matrix plus predict label for each cluster ----------------
################################################################
class AssoMatrixPlusLabel(Function):
    @staticmethod
    def forward(ctx, ks: int, offset: torch.Tensor, sp_offset: torch.Tensor, idx_c: torch.Tensor, lab: torch.Tensor, cid: torch.Tensor = None, category=13) -> Tuple[torch.Tensor]:
        """
        KNN Indexing
        input: ks: int32, Number of cluster neighbors of each point
               offset: (b) offset of each batch
               sp_offset: (b) offset of each superpoint
               idx_c: (n, ks) indexs of cluster
               lab: (n, 1) label of point
               cid: (m, 1) centriods
        output: idx: (m, n)
                cnt: (m, 1)
                clab: (m, classes)
        """
        assert idx_c.is_contiguous()
        assert cid.is_contiguous()
        m, _ = cid.size()
        n = idx_c.size(0)
        # print('category: {}'.format(category))
        idx = torch.cuda.IntTensor(m, n).zero_()
        cnt = torch.cuda.IntTensor(m, 1).zero_()
        clab = torch.cuda.IntTensor(m, category).zero_()
        pointops_sp_v2_cuda.assomatrix_label_cuda(n, m, ks, category, offset, sp_offset, idx_c, lab, cid, idx, cnt, clab)
        return idx, cnt, clab

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None, None, None, None

assomatrixpluslabel = AssoMatrixPlusLabel.apply


class Grouping(Function):
    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        input: features: (b, c, n), idx : (b, m, nsample) containing the indicies of features to group with
        output: (b, c, m, nsample)
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()
        b, c, n = features.size()
        _, m, nsample = idx.size()
        output = torch.cuda.FloatTensor(b, c, m, nsample)
        pointops_sp_v2_cuda.grouping_forward_cuda(b, c, n, m, nsample, features, idx, output)
        ctx.for_backwards = (idx, n)
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        input: grad_out: (b, c, m, nsample)
        output: (b, c, n), None
        """
        idx, n = ctx.for_backwards
        b, c, m, nsample = grad_out.size()
        grad_features = torch.cuda.FloatTensor(b, c, n).zero_()
        grad_out_data = grad_out.data.contiguous()
        pointops_sp_v2_cuda.grouping_backward_cuda(b, c, n, m, nsample, grad_out_data, idx, grad_features.data)
        return grad_features, None

grouping = Grouping.apply