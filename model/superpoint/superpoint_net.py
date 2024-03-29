import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import open3d as o3d
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../'))

from lib.pointops.functions import pointops
from lib.boundaryops.functions import boundaryops
from lib.pointops_sp.functions import pointops_sp
from lib.pointops_sp_v2.functions import pointops_sp_v2
from modules import learn_SLIC_calc_v1_new, init_fea, calc_sp_fea, \
    point_normal_similarity_loss, calc_sp_normal, learn_SLIC_calc_v2, calc_sp_fea_v2, \
    Contrastive_InfoNCE_loss_sp, Contrastive_InfoNCE_loss_p2sp, Contrastive_InfoNCE_loss_re_p_fea, infoNCE_loss_p2sp, \
    superpoint_transformer, superpoint_transformer_v2, LPE_stn_recurrent, superpoint_transformer_v3


class PointTransformerLayer(nn.Module):
    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        super().__init__()
        self.mid_planes = mid_planes = out_planes // 1
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample
        self.linear_q = nn.Linear(in_planes, mid_planes)
        self.linear_k = nn.Linear(in_planes, mid_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)
        self.linear_p = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3), nn.ReLU(inplace=True), nn.Linear(3, out_planes))
        self.linear_w = nn.Sequential(nn.BatchNorm1d(mid_planes), nn.ReLU(inplace=True),
                                    nn.Linear(mid_planes, mid_planes // share_planes),
                                    nn.BatchNorm1d(mid_planes // share_planes), nn.ReLU(inplace=True),
                                    nn.Linear(out_planes // share_planes, out_planes // share_planes))
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, pxo) -> torch.Tensor:
        p, x, o = pxo  # (n, 3), (n, c), (b)
        x_q, x_k, x_v = self.linear_q(x), self.linear_k(x), self.linear_v(x)  # (n, c)
        x_k = pointops.queryandgroup(self.nsample, p, p, x_k, None, o, o, use_xyz=True)  # (n, nsample, 3+c)
        x_v = pointops.queryandgroup(self.nsample, p, p, x_v, None, o, o, use_xyz=False)  # (n, nsample, c)
        p_r, x_k = x_k[:, :, 0:3], x_k[:, :, 3:]
        for i, layer in enumerate(self.linear_p): p_r = layer(p_r.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i == 1 else layer(p_r)    # (n, nsample, c)
        w = x_k - x_q.unsqueeze(1) + p_r.view(p_r.shape[0], p_r.shape[1], self.out_planes // self.mid_planes, self.mid_planes).sum(2)  # (n, nsample, c)
        for i, layer in enumerate(self.linear_w): w = layer(w.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i % 3 == 0 else layer(w)
        w = self.softmax(w)  # (n, nsample, c)
        n, nsample, c = x_v.shape; s = self.share_planes
        x = ((x_v + p_r).view(n, nsample, s, c // s) * w.unsqueeze(2)).sum(1).view(n, c)
        return x


def boundary_queryandgroup(nsample, xyz, new_xyz, feat, idx, offset, new_offset, edges, boundary, use_xyz=True):
    """
    input: xyz: (n, 3), new_xyz: (m, 3), feat: (n, c), idx: (m, nsample), offset: (b), new_offset: (b)
    output: new_feat: (m, c+3, nsample), grouped_idx: (m, nsample)
    """
    assert xyz.is_contiguous() and new_xyz.is_contiguous() and feat.is_contiguous()
    if new_xyz is None:
        new_xyz = xyz
    if idx is None:
        idx, _ = boundaryops.boundaryquery(nsample, xyz, new_xyz, offset, new_offset, edges, boundary)

        # idx, _ = pointops.knnquery(nsample, xyz, new_xyz, offset, new_offset) # (m, nsample)
        # idx = idx.cpu().numpy().tolist()
        # idx_1 = []
        # boundary = boundary.cpu().numpy()
        # for i in range(len(offset)):
        #     if i == 0:
        #         xyz_ = xyz[0:offset[i]]
        #         edges_ = edges[0:offset[i]]
        #     else:
        #         xyz_ = xyz[offset[i-1]:offset[i]]
        #         edges_ = edges[offset[i-1]:offset[i]]
        #     # edges = trimesh.geometry.faces_to_edges(F_.cpu().numpy())
        #     # edges = torch.tensor(edges).to(F_.device)

        #     for j in range(len(xyz_)):
        #         q = queue.Queue()
        #         q.put(j)
        #         n_p = []
        #         n_p.append(j)

        #         if boundary[j] == 1:
        #             n_p = idx[j]
        #         else:
        #             while(len(n_p) < nsample):
        #                 if q.qsize() != 0:
        #                     q_n = q.get()
        #                 else:
        #                     n_p = idx[j]
        #                     break
        #                 # n, _ = np.where(edges == q_n)
        #                 # nn_idx = np.unique(edges[n][edges[n] != q_n])
        #                 # nn_idx = nn_idx[boundary[nn_idx] == 0]
        #                 nn_idx = edges_[q_n][boundary[edges_[q_n]] == 0]
        #                 for nn in nn_idx:
        #                     if nn not in n_p:
        #                         q.put(nn)
        #                         n_p.append(nn)
        #                     if len(n_p) == nsample:
        #                         break
        #         # if type(n_p) != torch.Tensor:
        #         #     n_p = torch.tensor(n_p)
        #         idx_1.append(n_p)
        #         # del q
        # idx_1 = torch.tensor(idx_1)
        # idx = idx_1
                    
    n, m, c = xyz.shape[0], new_xyz.shape[0], feat.shape[1]
    grouped_xyz = xyz[idx.view(-1).long(), :].view(m, nsample, 3) # (m, nsample, 3)
    #grouped_xyz = grouping(xyz, idx) # (m, nsample, 3)
    grouped_xyz -= new_xyz.unsqueeze(1) # (m, nsample, 3)
    grouped_feat = feat[idx.view(-1).long(), :].view(m, nsample, c) # (m, nsample, c)
    #grouped_feat = grouping(feat, idx) # (m, nsample, c)

    if use_xyz:
        return torch.cat((grouped_xyz, grouped_feat), -1), idx # (m, nsample, 3+c)
    else:
        return grouped_feat, idx
    

class BoundaryTransformerLayer(nn.Module):
    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        super().__init__()
        self.mid_planes = mid_planes = out_planes // 1
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample
        self.linear_q = nn.Linear(in_planes, mid_planes)
        self.linear_k = nn.Linear(in_planes, mid_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)
        self.linear_p = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3), nn.ReLU(inplace=True), nn.Linear(3, out_planes))
        self.linear_w = nn.Sequential(nn.BatchNorm1d(mid_planes), nn.ReLU(inplace=True),
                                    nn.Linear(mid_planes, mid_planes // share_planes),
                                    nn.BatchNorm1d(mid_planes // share_planes), nn.ReLU(inplace=True),
                                    nn.Linear(out_planes // share_planes, out_planes // share_planes))
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, pxo, edges, boundary) -> torch.Tensor:
        p, x, o = pxo  # (n, 3), (n, c), (b)
        x_q, x_k, x_v = self.linear_q(x), self.linear_k(x), self.linear_v(x)  # (n, c)
        x_k, idx = boundary_queryandgroup(self.nsample, p, p, x_k, None, o, o, edges, boundary, use_xyz=True)  # (n, nsample, 3+c)
        x_v, idx = boundary_queryandgroup(self.nsample, p, p, x_v, None, o, o, edges, boundary, use_xyz=False)  # (n, nsample, c)
        # x_k = pointops.queryandgroup(self.nsample, p, p, x_k, None, o, o, use_xyz=True)  # (n, nsample, 3+c)
        # x_v = pointops.queryandgroup(self.nsample, p, p, x_v, None, o, o, use_xyz=False)  # (n, nsample, c)
        p_r, x_k = x_k[:, :, 0:3], x_k[:, :, 3:]
        for i, layer in enumerate(self.linear_p): p_r = layer(p_r.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i == 1 else layer(p_r)    # (n, nsample, c)
        w = x_k - x_q.unsqueeze(1) + p_r.view(p_r.shape[0], p_r.shape[1], self.out_planes // self.mid_planes, self.mid_planes).sum(2)  # (n, nsample, c)
        for i, layer in enumerate(self.linear_w): w = layer(w.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i % 3 == 0 else layer(w)
        w = self.softmax(w)  # (n, nsample, c)
        n, nsample, c = x_v.shape; s = self.share_planes
        x = ((x_v + p_r).view(n, nsample, s, c // s) * w.unsqueeze(2)).sum(1).view(n, c)
        return x

class BoundaryTransformerBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, share_planes=8, nsample=16):
        super(BoundaryTransformerBlock, self).__init__()
        self.linear1 = nn.Linear(in_planes, planes, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.transformer2 = BoundaryTransformerLayer(planes, planes, share_planes, nsample)
        self.bn2 = nn.BatchNorm1d(planes)
        self.linear3 = nn.Linear(planes, planes * self.expansion, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo, edges, boundary):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        identity = x
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.transformer2([p, x, o], edges, boundary)))
        x = self.bn3(self.linear3(x))
        x += identity
        x = self.relu(x)
        return [p, x, o]


class TransitionDown(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, nsample=16):
        super().__init__()
        self.stride, self.nsample = stride, nsample
        if stride != 1:
            self.linear = nn.Linear(3+in_planes, out_planes, bias=False)
            self.pool = nn.MaxPool1d(nsample)
        else:
            self.linear = nn.Linear(in_planes, out_planes, bias=False)
        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        if self.stride != 1:
            n_o, count = [o[0].item() // self.stride], o[0].item() // self.stride
            for i in range(1, o.shape[0]):
                count += (o[i].item() - o[i-1].item()) // self.stride
                n_o.append(count)
            n_o = torch.cuda.IntTensor(n_o)
            idx = pointops.furthestsampling(p, o, n_o)  # (m)
            n_p = p[idx.long(), :]  # (m, 3)
            x = pointops.queryandgroup(self.nsample, p, n_p, x, None, o, n_o, use_xyz=True)  # (m, 3+c, nsample)
            x = self.relu(self.bn(self.linear(x).transpose(1, 2).contiguous()))  # (m, c, nsample)
            x = self.pool(x).squeeze(-1)  # (m, c)
            p, o = n_p, n_o
        else:
            x = self.relu(self.bn(self.linear(x)))  # (n, c)
        return [p, x, o]


class TransitionUp(nn.Module):
    def __init__(self, in_planes, out_planes=None):
        super().__init__()
        if out_planes is None:
            self.linear1 = nn.Sequential(nn.Linear(2*in_planes, in_planes), nn.BatchNorm1d(in_planes), nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(nn.Linear(in_planes, in_planes), nn.ReLU(inplace=True))
        else:
            self.linear1 = nn.Sequential(nn.Linear(out_planes, out_planes), nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(nn.Linear(in_planes, out_planes), nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True))
        
    def forward(self, pxo1, pxo2=None):
        if pxo2 is None:
            _, x, o = pxo1  # (n, 3), (n, c), (b)
            x_tmp = []
            for i in range(o.shape[0]):
                if i == 0:
                    s_i, e_i, cnt = 0, o[0], o[0]
                else:
                    s_i, e_i, cnt = o[i-1], o[i], o[i] - o[i-1]
                x_b = x[s_i:e_i, :]
                x_b = torch.cat((x_b, self.linear2(x_b.sum(0, True) / cnt).repeat(cnt, 1)), 1)
                x_tmp.append(x_b)
            x = torch.cat(x_tmp, 0)
            x = self.linear1(x)
        else:
            p1, x1, o1 = pxo1; p2, x2, o2 = pxo2
            x = self.linear1(x1) + pointops.interpolation(p2, p1, self.linear2(x2), o2, o1)
        return x


class TransitionDown_v2(nn.Module):
    def __init__(self, in_channels, out_channels, ratio, k, norm_layer=nn.LayerNorm):
        super().__init__()
        self.ratio = ratio
        self.k = k
        self.norm = norm_layer(in_channels) if norm_layer else None
        if ratio != 1:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.pool = nn.MaxPool1d(k)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, pxo):
        xyz, feats, offset = pxo  # (n, 3), (n, c), (b)
        if self.ratio != 1:
            n_offset, count = [int(offset[0].item()*self.ratio)+1], int(offset[0].item()*self.ratio)+1
            for i in range(1, offset.shape[0]):
                count += ((offset[i].item() - offset[i-1].item())*self.ratio) + 1
                n_offset.append(count)
            n_offset = torch.cuda.IntTensor(n_offset)
            idx = pointops.furthestsampling(xyz, offset, n_offset)  # (m)
            n_xyz = xyz[idx.long(), :]  # (m, 3)

            feats = pointops.queryandgroup(self.k, xyz, n_xyz, feats, None, offset, n_offset, use_xyz=False)  # (m, nsample, 3+c)
            m, k, c = feats.shape
            feats = self.linear(self.norm(feats.view(m*k, c)).view(m, k, c)).transpose(1, 2).contiguous()
            feats = self.pool(feats).squeeze(-1)  # (m, c)
        else:
            feats = self.linear(self.norm(feats))
            n_xyz = xyz
            n_offset = offset
        
        return [n_xyz, feats, n_offset]


class Upsample(nn.Module):
    def __init__(self, k, in_channels, out_channels, bn_momentum=0.02):
        super().__init__()
        self.k = k
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.linear1 = nn.Sequential(nn.LayerNorm(out_channels), nn.Linear(out_channels, out_channels))
        self.linear2 = nn.Sequential(nn.LayerNorm(in_channels), nn.Linear(in_channels, out_channels))

    def forward(self, pxo1, pxo2):
        support_xyz, support_feats, support_offset = pxo1; xyz, feats, offset = pxo2
        feats = self.linear1(support_feats) + pointops.interpolation(xyz, support_xyz, self.linear2(feats), offset, support_offset)
        return feats


class PointTransformerBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, share_planes=8, nsample=16):
        super(PointTransformerBlock, self).__init__()
        self.linear1 = nn.Linear(in_planes, planes, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.transformer2 = PointTransformerLayer(planes, planes, share_planes, nsample)
        self.bn2 = nn.BatchNorm1d(planes)
        self.linear3 = nn.Linear(planes, planes * self.expansion, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        identity = x
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.transformer2([p, x, o])))
        x = self.bn3(self.linear3(x))
        x += identity
        x = self.relu(x)
        return [p, x, o]


class PointTransformerSeg(nn.Module):
    def __init__(self, block, blocks, c=6, k=13, args=None):
        super().__init__()
        self.c = c
        self.in_planes, planes = c, [32, 64, 128, 256, 512]
        fpn_planes, fpnhead_planes, share_planes = 128, 64, 8
        stride, nsample = [1, 4, 4, 4, 4], [8, 16, 16, 16, 16]
        self.enc1 = self._make_enc(block, planes[0], blocks[0], share_planes, stride=stride[0], nsample=nsample[0])  # N/1
        self.enc2 = self._make_enc(block, planes[1], blocks[1], share_planes, stride=stride[1], nsample=nsample[1])  # N/4
        self.enc3 = self._make_enc(block, planes[2], blocks[2], share_planes, stride=stride[2], nsample=nsample[2])  # N/16
        self.enc4 = self._make_enc(block, planes[3], blocks[3], share_planes, stride=stride[3], nsample=nsample[3])  # N/64
        self.enc5 = self._make_enc(block, planes[4], blocks[4], share_planes, stride=stride[4], nsample=nsample[4])  # N/256
        self.dec5 = self._make_dec(block, planes[4], 2, share_planes, nsample=nsample[4], is_head=True)  # transform p5
        self.dec4 = self._make_dec(block, planes[3], 2, share_planes, nsample=nsample[3])  # fusion p5 and p4
        self.dec3 = self._make_dec(block, planes[2], 2, share_planes, nsample=nsample[2])  # fusion p4 and p3
        self.dec2 = self._make_dec(block, planes[1], 2, share_planes, nsample=nsample[1])  # fusion p3 and p2
        self.dec1 = self._make_dec(block, planes[0], 2, share_planes, nsample=nsample[0])  # fusion p2 and p1
        self.cls = nn.Sequential(nn.Linear(planes[0], planes[0]), nn.BatchNorm1d(planes[0]), nn.ReLU(inplace=True), nn.Linear(planes[0], k))
        self.boundary = nn.Sequential(nn.Linear(planes[0], planes[0]), nn.BatchNorm1d(planes[0]), nn.ReLU(inplace=True), nn.Linear(planes[0], 2))
        self.embedding = nn.Sequential(nn.Linear(planes[0], planes[0]), nn.BatchNorm1d(planes[0]), nn.ReLU(inplace=True), nn.Linear(planes[0], planes[0]))

    def _make_enc(self, block, planes, blocks, share_planes=8, stride=1, nsample=16):
        layers = []
        layers.append(TransitionDown(self.in_planes, planes * block.expansion, stride, nsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)

    def _make_dec(self, block, planes, blocks, share_planes=8, nsample=16, is_head=False):
        layers = []
        layers.append(TransitionUp(self.in_planes, None if is_head else planes * block.expansion))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)

    def forward(self, pxo):
        p0, x0, o0 = pxo  # (n, 3), (n, c), (b)
        x0 = p0 if self.c == 3 else torch.cat((p0, x0), 1)
        p1, x1, o1 = self.enc1([p0, x0, o0])
        p2, x2, o2 = self.enc2([p1, x1, o1])
        p3, x3, o3 = self.enc3([p2, x2, o2])
        p4, x4, o4 = self.enc4([p3, x3, o3])
        p5, x5, o5 = self.enc5([p4, x4, o4])
        x5 = self.dec5[1:]([p5, self.dec5[0]([p5, x5, o5]), o5])[1]
        x4 = self.dec4[1:]([p4, self.dec4[0]([p4, x4, o4], [p5, x5, o5]), o4])[1]
        x3 = self.dec3[1:]([p3, self.dec3[0]([p3, x3, o3], [p4, x4, o4]), o3])[1]
        x2 = self.dec2[1:]([p2, self.dec2[0]([p2, x2, o2], [p3, x3, o3]), o2])[1]
        x1 = self.dec1[1:]([p1, self.dec1[0]([p1, x1, o1], [p2, x2, o2]), o1])[1]
        type_per_point = self.cls(x1)
        primitive_embedding = self.embedding(x1)
        boundary_pred = self.boundary(x1)

        return primitive_embedding, type_per_point, boundary_pred


def pointtransformer_seg_repro(**kwargs):
    model = PointTransformerSeg(PointTransformerBlock, [2, 3, 4, 6, 3], **kwargs)
    return model

class PointTransformer_Encoder(nn.Module):
    
    def __init__(self, input_channels=6, args=None):
        super(PointTransformer_Encoder, self).__init__()

        self.nsamples = args.get('nsamples', [8, 16, 16, 16, 16])
        self.strides = args.get('strides', [None, 4, 4, 4, 4])
        self.planes = args.get('planes', [64, 128, 256, 512])
        self.blocks = args.get('blocks', [2, 3, 4, 6, 3])
        self.c = input_channels

        # encoder
        self.in_mlp = nn.Sequential(
            nn.Linear(input_channels, self.planes[0], bias=False),
            nn.BatchNorm1d(self.planes[0]),
            nn.ReLU(inplace=True),
            nn.Linear(self.planes[0], self.planes[0], bias=False),
            nn.BatchNorm1d(self.planes[0]),
            nn.ReLU(inplace=True)
        )

        self.PtBlock1 = PointTransformerBlock(self.planes[0], self.planes[0], nsample=8)
        self.PtBlock2 = PointTransformerBlock(self.planes[0], self.planes[0], nsample=8)
        self.PtBlock3 = PointTransformerBlock(self.planes[0], self.planes[0], nsample=8)
        
        self.mlp1 = nn.Linear(256, 1024)
        self.bnmlp1 = nn.BatchNorm1d(1024)

    # def _break_up_pc(self, pc):
    #     xyz = pc[..., 0:3].contiguous().unsqueeze(dim=0)
    #     features = pc[..., 3:].contiguous().unsqueeze(dim=0) if pc.size(-1) > 3 else deepcopy(xyz)
    #     return xyz, features

    def forward(self, pxo):
        p0, x0, o0 = pxo  # (n, 3), (n, c), (b)
        x0 = p0 if self.c == 3 else torch.cat((p0, x0), 1)
        
        # encoder
        x0 = self.in_mlp(x0)

        p1, x1, o1 = self.PtBlock1([p0, x0, o0])
        p2, x2, o2 = self.PtBlock2([p1, x1, o1])
        p3, x3, o3 = self.PtBlock3([p2, x2, o2])

        x_features = torch.cat((x0, x1, x2, x3), dim=1)
        x = F.relu(self.bnmlp1(self.mlp1(x_features)))
        # x4 = x.max(dim=2)[0]

        return x, x_features

class PointTransformer_PrimSeg(nn.Module):

    def __init__(self, c=6, emb_size=128, k=10, primitives=True, embedding=True, param=True, mode=5, use_boundary=True, num_channels=6, nn_nb=80, args=None):
        super(PointTransformer_PrimSeg, self).__init__()

        self.encoder = PointTransformer_Encoder(input_channels=num_channels, args=args)
        self.drop = 0.0

        self.conv1 = torch.nn.Linear(1024 + 256, 512)

        self.bn1 = nn.BatchNorm1d(512)
        self.conv2 = torch.nn.Linear(512, 256)

        self.bn2 = nn.BatchNorm1d(256)

        self.softmax = torch.nn.Softmax(dim=1)
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        self.tanh = torch.nn.Tanh()
        self.emb_size = emb_size
        self.primitives = primitives
        self.embedding = embedding
        self.param = param
        self.use_boundary = use_boundary

        # self.pt_layer = PointTransformerBlock(512)

        if self.embedding:
            self.mlp_seg_prob1 = torch.nn.Linear(256, 256)
            self.mlp_seg_prob2 = torch.nn.Linear(256, self.emb_size)
            self.bn_seg_prob1 = nn.BatchNorm1d(256)

        if primitives:
            self.mlp_prim_prob1 = torch.nn.Linear(256, 256)
            self.mlp_prim_prob2 = torch.nn.Linear(256, k)
            self.bn_prim_prob1 = nn.BatchNorm1d(256)
        
        # if param:
        #     self.mlp_param_prob1 = torch.nn.Conv1d(256, 256, 1)
        #     self.mlp_param_prob2 = torch.nn.Conv1d(256, 22, 1)
        #     self.bn_param_prob1 = nn.GroupNorm(4, 256)

        # if self.mode == 5:
        #     self.mlp_normal_prob1 = torch.nn.Conv1d(256, 256, 1)
        #     self.mlp_normal_prob2 = torch.nn.Conv1d(256, 3, 1)
        #     self.bn_normal_prob1 = nn.GroupNorm(4, 256)
        
        if use_boundary:
            self.mlp_bound_prob1 = torch.nn.Linear(256, 256)
            self.mlp_bound_prob2 = torch.nn.Linear(256, 2)
            self.bn_bound_prob1 = nn.BatchNorm1d(256)

    def forward(self, pxo):

        x, first_layer_features = self.encoder(pxo)

        x = torch.cat([x, first_layer_features], 1)

        x = F.dropout(F.relu(self.bn1(self.conv1(x))), self.drop)
        x_all = F.dropout(F.relu(self.bn2(self.conv2(x))), self.drop)
        if self.embedding:
            x = F.dropout(F.relu(self.bn_seg_prob1(self.mlp_seg_prob1(x_all))), self.drop)
            embedding = self.mlp_seg_prob2(x)

        if self.primitives:
            x = F.dropout(F.relu(self.bn_prim_prob1(self.mlp_prim_prob1(x_all))), self.drop)
            type_per_point = self.mlp_prim_prob2(x)

            # type_per_point = self.logsoftmax(type_per_point).permute(0, 2, 1)
        
        # if self.param:
        #     x = F.dropout(F.relu(self.bn_param_prob1(self.mlp_param_prob1(x_all))), self.drop)
        #     param_per_point = self.mlp_param_prob2(x).transpose(1, 2)
        #     sphere_param = param_per_point[:, :, :4]
        #     plane_norm = torch.norm(param_per_point[:, :, 4:7], dim=-1, keepdim=True).repeat(1, 1, 3) + 1e-12
        #     plane_normal = param_per_point[:, :, 4:7] / plane_norm
        #     plane_param = torch.cat([plane_normal, param_per_point[:,:,7:8]], dim=2)
        #     cylinder_norm = torch.norm(param_per_point[:, :, 8:11], dim=-1, keepdim=True).repeat(1, 1, 3) + 1e-12
        #     cylinder_normal = param_per_point[:, :, 8:11] / cylinder_norm
        #     cylinder_param = torch.cat([cylinder_normal, param_per_point[:, :, 11:15]], dim=2)

        #     cone_norm = torch.norm(param_per_point[:, :, 15:18], dim=-1, keepdim=True).repeat(1, 1, 3) + 1e-12
        #     cone_normal = param_per_point[:, :, 15:18] / cone_norm
        #     cone_param = torch.cat([cone_normal, param_per_point[:, :, 18:22]], dim=2)

        #     param_per_point = torch.cat([sphere_param, plane_param, cylinder_param, cone_param], dim=2)

        if self.use_boundary:
            x = F.dropout(F.relu(self.bn_bound_prob1(self.mlp_bound_prob1(x_all))), self.drop)
            boundary_per_point = self.mlp_bound_prob2(x)

            
        return embedding, type_per_point, boundary_per_point

class BoundaryTransformer_Encoder(nn.Module):
    
    def __init__(self, input_channels=6, args=None):
        super(BoundaryTransformer_Encoder, self).__init__()

        self.nsamples = args.get('nsamples', [8, 16, 16, 16, 16])
        self.strides = args.get('strides', [None, 4, 4, 4, 4])
        self.planes = args.get('planes', [64, 128, 256, 512])
        self.blocks = args.get('blocks', [2, 3, 4, 6, 3])
        self.c = input_channels

        # encoder
        self.in_mlp = nn.Sequential(
            nn.Linear(input_channels, self.planes[0], bias=False),
            nn.BatchNorm1d(self.planes[0]),
            nn.ReLU(inplace=True),
            nn.Linear(self.planes[0], self.planes[0], bias=False),
            nn.BatchNorm1d(self.planes[0]),
            nn.ReLU(inplace=True)
        )

        self.PtBlock1 = BoundaryTransformerBlock(self.planes[0], self.planes[0], nsample=8)
        self.PtBlock2 = BoundaryTransformerBlock(self.planes[0], self.planes[0], nsample=8)
        self.PtBlock3 = BoundaryTransformerBlock(self.planes[0], self.planes[0], nsample=8)
        
        self.mlp1 = nn.Linear(256, 1024)
        self.bnmlp1 = nn.BatchNorm1d(1024)

    # def _break_up_pc(self, pc):
    #     xyz = pc[..., 0:3].contiguous().unsqueeze(dim=0)
    #     features = pc[..., 3:].contiguous().unsqueeze(dim=0) if pc.size(-1) > 3 else deepcopy(xyz)
    #     return xyz, features

    def forward(self, pxo, edges, boundary):
        p0, x0, o0 = pxo  # (n, 3), (n, c), (b)
        x0 = p0 if self.c == 3 else torch.cat((p0, x0), 1)
        
        # encoder
        x0 = self.in_mlp(x0)

        p1, x1, o1 = self.PtBlock1([p0, x0, o0], edges, boundary)
        p2, x2, o2 = self.PtBlock2([p1, x1, o1], edges, boundary)
        p3, x3, o3 = self.PtBlock3([p2, x2, o2], edges, boundary)

        x_features = torch.cat((x0, x1, x2, x3), dim=1)
        x = F.relu(self.bnmlp1(self.mlp1(x_features)))
        # x4 = x.max(dim=2)[0]

        return x, x_features

class BoundaryTransformer_PrimSeg(nn.Module):

    def __init__(self, c=6, emb_size=128, k=10, primitives=True, embedding=True, param=True, mode=5, use_boundary=True, num_channels=6, nn_nb=80, args=None):
        super(BoundaryTransformer_PrimSeg, self).__init__()

        self.encoder = BoundaryTransformer_Encoder(input_channels=num_channels, args=args)
        self.drop = 0.0

        self.conv1 = torch.nn.Linear(1024 + 256, 512)

        self.bn1 = nn.BatchNorm1d(512)
        self.conv2 = torch.nn.Linear(512, 256)

        self.bn2 = nn.BatchNorm1d(256)

        self.softmax = torch.nn.Softmax(dim=1)
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        self.tanh = torch.nn.Tanh()
        self.emb_size = emb_size
        self.primitives = primitives
        self.embedding = embedding
        self.param = param
        self.use_boundary = use_boundary

        # self.pt_layer = PointTransformerBlock(512)

        if self.embedding:
            self.mlp_seg_prob1 = torch.nn.Linear(256, 256)
            self.mlp_seg_prob2 = torch.nn.Linear(256, self.emb_size)
            self.bn_seg_prob1 = nn.BatchNorm1d(256)

        if primitives:
            self.mlp_prim_prob1 = torch.nn.Linear(256, 256)
            self.mlp_prim_prob2 = torch.nn.Linear(256, k)
            self.bn_prim_prob1 = nn.BatchNorm1d(256)
        
        # if param:
        #     self.mlp_param_prob1 = torch.nn.Conv1d(256, 256, 1)
        #     self.mlp_param_prob2 = torch.nn.Conv1d(256, 22, 1)
        #     self.bn_param_prob1 = nn.GroupNorm(4, 256)

        # if self.mode == 5:
        #     self.mlp_normal_prob1 = torch.nn.Conv1d(256, 256, 1)
        #     self.mlp_normal_prob2 = torch.nn.Conv1d(256, 3, 1)
        #     self.bn_normal_prob1 = nn.GroupNorm(4, 256)
        
        if use_boundary:
            self.mlp_bound_prob1 = torch.nn.Linear(256, 256)
            self.mlp_bound_prob2 = torch.nn.Linear(256, 2)
            self.bn_bound_prob1 = nn.BatchNorm1d(256)

    def forward(self, pxo, edges, boundary):

        x, first_layer_features = self.encoder(pxo, edges, boundary)

        x = torch.cat([x, first_layer_features], 1)

        x = F.dropout(F.relu(self.bn1(self.conv1(x))), self.drop)
        x_all = F.dropout(F.relu(self.bn2(self.conv2(x))), self.drop)
        if self.embedding:
            x = F.dropout(F.relu(self.bn_seg_prob1(self.mlp_seg_prob1(x_all))), self.drop)
            embedding = self.mlp_seg_prob2(x)

        if self.primitives:
            x = F.dropout(F.relu(self.bn_prim_prob1(self.mlp_prim_prob1(x_all))), self.drop)
            type_per_point = self.mlp_prim_prob2(x)

            # type_per_point = self.logsoftmax(type_per_point).permute(0, 2, 1)
        
        # if self.param:
        #     x = F.dropout(F.relu(self.bn_param_prob1(self.mlp_param_prob1(x_all))), self.drop)
        #     param_per_point = self.mlp_param_prob2(x).transpose(1, 2)
        #     sphere_param = param_per_point[:, :, :4]
        #     plane_norm = torch.norm(param_per_point[:, :, 4:7], dim=-1, keepdim=True).repeat(1, 1, 3) + 1e-12
        #     plane_normal = param_per_point[:, :, 4:7] / plane_norm
        #     plane_param = torch.cat([plane_normal, param_per_point[:,:,7:8]], dim=2)
        #     cylinder_norm = torch.norm(param_per_point[:, :, 8:11], dim=-1, keepdim=True).repeat(1, 1, 3) + 1e-12
        #     cylinder_normal = param_per_point[:, :, 8:11] / cylinder_norm
        #     cylinder_param = torch.cat([cylinder_normal, param_per_point[:, :, 11:15]], dim=2)

        #     cone_norm = torch.norm(param_per_point[:, :, 15:18], dim=-1, keepdim=True).repeat(1, 1, 3) + 1e-12
        #     cone_normal = param_per_point[:, :, 15:18] / cone_norm
        #     cone_param = torch.cat([cone_normal, param_per_point[:, :, 18:22]], dim=2)

        #     param_per_point = torch.cat([sphere_param, plane_param, cylinder_param, cone_param], dim=2)

        # if self.use_boundary:
        #     x = F.dropout(F.relu(self.bn_bound_prob1(self.mlp_bound_prob1(x_all))), self.drop)
        #     boundary_per_point = self.mlp_bound_prob2(x)

            
        return embedding, type_per_point
    

class BoundaryNet(nn.Module):

    def __init__(self, c=6, emb_size=128, k=10, primitives=True, embedding=True, param=True, mode=5, use_boundary=True, num_channels=6, nn_nb=80, args=None):
        super(BoundaryNet, self).__init__()

        self.encoder = PointTransformer_Encoder(input_channels=num_channels, args=args)
        self.drop = 0.0

        self.conv1 = torch.nn.Linear(1024 + 256, 512)

        self.bn1 = nn.BatchNorm1d(512)
        self.conv2 = torch.nn.Linear(512, 256)

        self.bn2 = nn.BatchNorm1d(256)

        self.softmax = torch.nn.Softmax(dim=1)
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        self.tanh = torch.nn.Tanh()
        self.emb_size = emb_size
        self.primitives = primitives
        self.embedding = embedding
        self.param = param
        self.use_boundary = use_boundary

        # self.pt_layer = PointTransformerBlock(512)

        if self.embedding:
            self.mlp_seg_prob1 = torch.nn.Linear(256, 256)
            self.mlp_seg_prob2 = torch.nn.Linear(256, self.emb_size)
            self.bn_seg_prob1 = nn.BatchNorm1d(256)

        if primitives:
            self.mlp_prim_prob1 = torch.nn.Linear(256, 256)
            self.mlp_prim_prob2 = torch.nn.Linear(256, k)
            self.bn_prim_prob1 = nn.BatchNorm1d(256)
        
        # if param:
        #     self.mlp_param_prob1 = torch.nn.Conv1d(256, 256, 1)
        #     self.mlp_param_prob2 = torch.nn.Conv1d(256, 22, 1)
        #     self.bn_param_prob1 = nn.GroupNorm(4, 256)

        # if self.mode == 5:
        #     self.mlp_normal_prob1 = torch.nn.Conv1d(256, 256, 1)
        #     self.mlp_normal_prob2 = torch.nn.Conv1d(256, 3, 1)
        #     self.bn_normal_prob1 = nn.GroupNorm(4, 256)
        
        if use_boundary:
            self.mlp_bound_prob1 = torch.nn.Linear(256, 256)
            self.mlp_bound_prob2 = torch.nn.Linear(256, 2)
            self.bn_bound_prob1 = nn.BatchNorm1d(256)

    def forward(self, pxo):

        x, first_layer_features = self.encoder(pxo)

        x = torch.cat([x, first_layer_features], 1)

        x = F.dropout(F.relu(self.bn1(self.conv1(x))), self.drop)
        x_all = F.dropout(F.relu(self.bn2(self.conv2(x))), self.drop)
        # if self.embedding:
        #     x = F.dropout(F.relu(self.bn_seg_prob1(self.mlp_seg_prob1(x_all))), self.drop)
        #     embedding = self.mlp_seg_prob2(x)

        # if self.primitives:
        #     x = F.dropout(F.relu(self.bn_prim_prob1(self.mlp_prim_prob1(x_all))), self.drop)
        #     type_per_point = self.mlp_prim_prob2(x)

            # type_per_point = self.logsoftmax(type_per_point).permute(0, 2, 1)
        
        # if self.param:
        #     x = F.dropout(F.relu(self.bn_param_prob1(self.mlp_param_prob1(x_all))), self.drop)
        #     param_per_point = self.mlp_param_prob2(x).transpose(1, 2)
        #     sphere_param = param_per_point[:, :, :4]
        #     plane_norm = torch.norm(param_per_point[:, :, 4:7], dim=-1, keepdim=True).repeat(1, 1, 3) + 1e-12
        #     plane_normal = param_per_point[:, :, 4:7] / plane_norm
        #     plane_param = torch.cat([plane_normal, param_per_point[:,:,7:8]], dim=2)
        #     cylinder_norm = torch.norm(param_per_point[:, :, 8:11], dim=-1, keepdim=True).repeat(1, 1, 3) + 1e-12
        #     cylinder_normal = param_per_point[:, :, 8:11] / cylinder_norm
        #     cylinder_param = torch.cat([cylinder_normal, param_per_point[:, :, 11:15]], dim=2)

        #     cone_norm = torch.norm(param_per_point[:, :, 15:18], dim=-1, keepdim=True).repeat(1, 1, 3) + 1e-12
        #     cone_normal = param_per_point[:, :, 15:18] / cone_norm
        #     cone_param = torch.cat([cone_normal, param_per_point[:, :, 18:22]], dim=2)

        #     param_per_point = torch.cat([sphere_param, plane_param, cylinder_param, cone_param], dim=2)

        if self.use_boundary:
            x = F.dropout(F.relu(self.bn_bound_prob1(self.mlp_bound_prob1(x_all))), self.drop)
            boundary_per_point = self.mlp_bound_prob2(x)

            
        return boundary_per_point


class PointTransformer_Unet_PrimSeg(nn.Module):
    def __init__(self, block, blocks, c=6, k=13, args=None):
        super().__init__()
        self.c = c
        self.in_planes, planes = c, [32, 64, 128, 256, 512]
        fpn_planes, fpnhead_planes, share_planes = 128, 64, 8
        stride, nsample = [1, 0.25, 0.25, 0.25, 0.25], [8, 16, 16, 16, 16]
        self.enc1 = self._make_enc(block, planes[0], blocks[0], share_planes, stride=stride[0], nsample=nsample[0])  # N/1
        self.enc2 = self._make_enc(block, planes[1], blocks[1], share_planes, stride=stride[1], nsample=nsample[1])  # N/4
        self.enc3 = self._make_enc(block, planes[2], blocks[2], share_planes, stride=stride[2], nsample=nsample[2])  # N/16
        self.enc4 = self._make_enc(block, planes[3], blocks[3], share_planes, stride=stride[3], nsample=nsample[3])  # N/64
        self.enc5 = self._make_enc(block, planes[4], blocks[4], share_planes, stride=stride[4], nsample=nsample[4])  # N/256
        # self.dec5 = self._make_dec(block, planes[4], 2, share_planes, nsample=nsample[4], is_head=True)  # transform p5
        self.dec4 = self._make_dec(block, planes[3], 2, share_planes, nsample=nsample[3])  # fusion p5 and p4
        self.dec3 = self._make_dec(block, planes[2], 2, share_planes, nsample=nsample[2])  # fusion p4 and p3
        self.dec2 = self._make_dec(block, planes[1], 2, share_planes, nsample=nsample[1])  # fusion p3 and p2
        self.dec1 = self._make_dec(block, planes[0], 2, share_planes, nsample=nsample[0])  # fusion p2 and p1
        self.cls = nn.Sequential(nn.Linear(planes[0], planes[0]), nn.BatchNorm1d(planes[0]), nn.ReLU(inplace=True), nn.Linear(planes[0], k))
        self.boundary = nn.Sequential(nn.Linear(planes[0], planes[0]), nn.BatchNorm1d(planes[0]), nn.ReLU(inplace=True), nn.Linear(planes[0], 2))
        self.embedding = nn.Sequential(nn.Linear(planes[0], planes[0]), nn.BatchNorm1d(planes[0]), nn.ReLU(inplace=True), nn.Linear(planes[0], planes[0]))

    def _make_enc(self, block, planes, blocks, share_planes=8, stride=1, nsample=16):
        layers = []
        layers.append(TransitionDown_v2(self.in_planes, planes * block.expansion, ratio=stride, k=nsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)
    
    def _make_dec(self, block, planes, blocks, share_planes=8, nsample=16, is_head=False):
        layers = []
        layers.append(Upsample(nsample, self.in_planes, None if is_head else planes * block.expansion))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)
    
    def forward(self, pxo):
        p0, x0, o0 = pxo  # (n, 3), (n, c), (b)
        x0 = p0 if self.c == 3 else torch.cat((p0, x0), 1)
        p1, x1, o1 = self.enc1([p0, x0, o0])
        p2, x2, o2 = self.enc2([p1, x1, o1])
        p3, x3, o3 = self.enc3([p2, x2, o2])
        p4, x4, o4 = self.enc4([p3, x3, o3])
        p5, x5, o5 = self.enc5([p4, x4, o4])
        # x5 = self.dec5[1:]([p5, self.dec5[0]([p5, x5, o5]), o5])[1]
        x4 = self.dec4[1:]([p4, self.dec4[0]([p4, x4, o4], [p5, x5, o5]), o4])[1]
        x3 = self.dec3[1:]([p3, self.dec3[0]([p3, x3, o3], [p4, x4, o4]), o3])[1]
        x2 = self.dec2[1:]([p2, self.dec2[0]([p2, x2, o2], [p3, x3, o3]), o2])[1]
        x1 = self.dec1[1:]([p1, self.dec1[0]([p1, x1, o1], [p2, x2, o2]), o1])[1]
        type_per_point = self.cls(x1)
        primitive_embedding = self.embedding(x1)
        boundary_pred = self.boundary(x1)

        return primitive_embedding, type_per_point, boundary_pred

def pointtransformer_Unit_seg_repro(**kwargs):
    model = PointTransformer_Unet_PrimSeg(PointTransformerBlock, [2, 3, 4, 6, 3], **kwargs)
    return model


class BoundaryPointTransformer_Unet_PrimSeg(nn.Module):
    def __init__(self, block, blocks, c=6, k=13, args=None):
        super().__init__()
        self.c = c
        self.in_planes, planes = c, [32, 64, 128, 256, 512]
        fpn_planes, fpnhead_planes, share_planes = 128, 64, 8
        stride, nsample = [1, 0.25, 0.25, 0.25, 0.25], [8, 16, 16, 16, 16]
        self.enc1 = self._make_enc(block, planes[0], blocks[0], share_planes, stride=stride[0], nsample=nsample[0])  # N/1
        self.enc2 = self._make_enc(block, planes[1], blocks[1], share_planes, stride=stride[1], nsample=nsample[1])  # N/4
        self.enc3 = self._make_enc(block, planes[2], blocks[2], share_planes, stride=stride[2], nsample=nsample[2])  # N/16
        self.enc4 = self._make_enc(block, planes[3], blocks[3], share_planes, stride=stride[3], nsample=nsample[3])  # N/64
        self.enc5 = self._make_enc(block, planes[4], blocks[4], share_planes, stride=stride[4], nsample=nsample[4])  # N/256
        # self.dec5 = self._make_dec(block, planes[4], 2, share_planes, nsample=nsample[4], is_head=True)  # transform p5
        self.dec4 = self._make_dec(block, planes[3], 2, share_planes, nsample=nsample[3])  # fusion p5 and p4
        self.dec3 = self._make_dec(block, planes[2], 2, share_planes, nsample=nsample[2])  # fusion p4 and p3
        self.dec2 = self._make_dec(block, planes[1], 2, share_planes, nsample=nsample[1])  # fusion p3 and p2
        self.dec1 = self._make_dec(block, planes[0], 2, share_planes, nsample=nsample[0])  # fusion p2 and p1
        self.cls = nn.Sequential(nn.Linear(planes[0], planes[0]), nn.BatchNorm1d(planes[0]), nn.ReLU(inplace=True), nn.Linear(planes[0], k))
        # self.boundary = nn.Sequential(nn.Linear(planes[0], planes[0]), nn.BatchNorm1d(planes[0]), nn.ReLU(inplace=True), nn.Linear(planes[0], 2))
        self.embedding = nn.Sequential(nn.Linear(planes[0], planes[0]), nn.BatchNorm1d(planes[0]), nn.ReLU(inplace=True), nn.Linear(planes[0], planes[0]))

    def _make_enc(self, block, planes, blocks, share_planes=8, stride=1, nsample=16):
        layers = []
        layers.append(TransitionDown_v2(self.in_planes, planes * block.expansion, ratio=stride, k=nsample))
        self.in_planes = planes * block.expansion
        if planes == 32:    # 首层添加Boundary采样
            block = BoundaryTransformerBlock
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)
    
    def _make_dec(self, block, planes, blocks, share_planes=8, nsample=16, is_head=False):
        layers = []
        layers.append(Upsample(nsample, self.in_planes, None if is_head else planes * block.expansion))
        self.in_planes = planes * block.expansion
        if planes == 32:
            block = BoundaryTransformerBlock
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)
    
    def forward(self, pxo, edges, boundary):
        p0, x0, o0 = pxo  # (n, 3), (n, c), (b)
        x0 = p0 if self.c == 3 else torch.cat((p0, x0), 1)
        # p1, x1, o1 = self.enc1([p0, x0, o0])
        p1, x1, o1 = self.enc1[1](self.enc1[0]([p0, x0, o0]), edges, boundary)
        p2, x2, o2 = self.enc2([p1, x1, o1])
        p3, x3, o3 = self.enc3([p2, x2, o2])
        p4, x4, o4 = self.enc4([p3, x3, o3])
        p5, x5, o5 = self.enc5([p4, x4, o4])
        # x5 = self.dec5[1:]([p5, self.dec5[0]([p5, x5, o5]), o5])[1]
        x4 = self.dec4[1:]([p4, self.dec4[0]([p4, x4, o4], [p5, x5, o5]), o4])[1]
        x3 = self.dec3[1:]([p3, self.dec3[0]([p3, x3, o3], [p4, x4, o4]), o3])[1]
        x2 = self.dec2[1:]([p2, self.dec2[0]([p2, x2, o2], [p3, x3, o3]), o2])[1]
        # x1 = self.dec1[1:]([p1, self.dec1[0]([p1, x1, o1], [p2, x2, o2]), o1])[1]
        x1 = self.dec1[1]([p1, self.dec1[0]([p1, x1, o1], [p2, x2, o2]), o1], edges, boundary)[1]
        type_per_point = self.cls(x1)
        primitive_embedding = self.embedding(x1)
        # boundary_pred = self.boundary(x1)

        return primitive_embedding, type_per_point

def boundarypointtransformer_Unit_seg_repro(**kwargs):
    model = BoundaryPointTransformer_Unet_PrimSeg(PointTransformerBlock, [2, 3, 4, 6, 3], **kwargs)
    return model


# 超点生成网络
class SuperPointNet(nn.Module):
    def __init__(self, block, blocks, c=6, k=13, args=None):
        super().__init__()
        self.c = c
        self.classes = k
        self.rate = args.rate
        self.in_planes, planes = c, [32, 64, 128, 256, 512]
        fpn_planes, fpnhead_planes, share_planes = 128, 64, 8
        stride, nsample = [1, 4, 4, 4, 4], [8, 16, 16, 16, 16]
        self.enc1 = self._make_enc(block, planes[0], blocks[0], share_planes, stride=stride[0], nsample=nsample[0])  # N/1
        self.enc2 = self._make_enc(block, planes[1], blocks[1], share_planes, stride=stride[1], nsample=nsample[1])  # N/4
        self.enc3 = self._make_enc(block, planes[2], blocks[2], share_planes, stride=stride[2], nsample=nsample[2])  # N/16
        self.enc4 = self._make_enc(block, planes[3], blocks[3], share_planes, stride=stride[3], nsample=nsample[3])  # N/64
        self.enc5 = self._make_enc(block, planes[4], blocks[4], share_planes, stride=stride[4], nsample=nsample[4])  # N/256
        self.dec5 = self._make_dec(block, planes[4], 2, share_planes, nsample=nsample[4], is_head=True)  # transform p5
        self.dec4 = self._make_dec(block, planes[3], 2, share_planes, nsample=nsample[3])  # fusion p5 and p4
        self.dec3 = self._make_dec(block, planes[2], 2, share_planes, nsample=nsample[2])  # fusion p4 and p3
        self.dec2 = self._make_dec(block, planes[1], 2, share_planes, nsample=nsample[1])  # fusion p3 and p2
        self.dec1 = self._make_dec(block, planes[0], 2, share_planes, nsample=nsample[0])  # fusion p2 and p1
        self.cls = nn.Sequential(nn.Linear(planes[0], planes[0]), nn.BatchNorm1d(planes[0]), nn.ReLU(inplace=True), nn.Linear(planes[0], k))
        self.boundary = nn.Sequential(nn.Linear(planes[0], planes[0]), nn.BatchNorm1d(planes[0]), nn.ReLU(inplace=True), nn.Linear(planes[0], 2))
        self.embedding = nn.Sequential(nn.Linear(planes[0], planes[0]), nn.BatchNorm1d(planes[0]), nn.ReLU(inplace=True), nn.Linear(planes[0], planes[0]))
        self.embedding64 = nn.Sequential(nn.Linear(planes[0], planes[0]), nn.BatchNorm1d(planes[0]), nn.ReLU(inplace=True), nn.Linear(planes[0], 64))
        # self.spmlp = nn.Sequential(nn.Linear(planes[0], planes[0]), nn.BatchNorm1d(planes[0]), nn.ReLU(inplace=True), nn.Linear(planes[0], planes[0]))
        self.parameter = nn.Sequential(nn.Linear(planes[0], planes[0]), nn.BatchNorm1d(planes[0]), nn.ReLU(inplace=True), nn.Linear(planes[0], 22))

        # self.learn_SLIC_calc_1 = learn_SLIC_calc_v1_new(ch_wc2p_fea=[32, 16, 16], ch_wc2p_xyz=[3, 16, 16], ch_mlp=[32, 16, 16],
        #                     bn=True, use_xyz=True, use_softmax=True, use_norm=False)

        # self.learn_SLIC_calc_2 = learn_SLIC_calc_v1_new(ch_wc2p_fea=[32, 16, 16], ch_wc2p_xyz=[3, 16, 16], ch_mlp=[32, 16, 16],
        #                     bn=True, use_xyz=True, use_softmax=True, use_norm=False)

        # self.learn_SLIC_calc_3 = learn_SLIC_calc_v1_new(ch_wc2p_fea=[32, 16, 16], ch_wc2p_xyz=[3, 16, 16], ch_mlp=[32, 16, 16],
        #                     bn=True, use_xyz=True, use_softmax=True, use_norm=False)

        # self.learn_SLIC_calc_4 = learn_SLIC_calc_v1_new(ch_wc2p_fea=[32, 16, 16], ch_wc2p_xyz=[3, 16, 16], ch_mlp=[32, 16, 16],
        #                     bn=True, use_xyz=True, use_softmax=True, use_norm=False, last=True)

        self.learn_SLIC_calc_1 = learn_SLIC_calc_v2(ch_wc2p_fea=[32, 16, 16], ch_wc2p_xyz=[3, 16, 16], ch_mlp=[32, 16, 16],
                            bn=True, use_xyz=True, use_softmax=True, use_norm=False)

        self.learn_SLIC_calc_2 = learn_SLIC_calc_v2(ch_wc2p_fea=[32, 16, 16], ch_wc2p_xyz=[3, 16, 16], ch_mlp=[32, 16, 16],
                            bn=True, use_xyz=True, use_softmax=True, use_norm=False)

        self.learn_SLIC_calc_3 = learn_SLIC_calc_v2(ch_wc2p_fea=[32, 16, 16], ch_wc2p_xyz=[3, 16, 16], ch_mlp=[32, 16, 16],
                            bn=True, use_xyz=True, use_softmax=True, use_norm=False)

        self.learn_SLIC_calc_4 = learn_SLIC_calc_v2(ch_wc2p_fea=[32, 16, 16], ch_wc2p_xyz=[3, 16, 16], ch_mlp=[32, 16, 16],
                            bn=True, use_xyz=True, use_softmax=True, use_norm=False, last=True)

    def _make_enc(self, block, planes, blocks, share_planes=8, stride=1, nsample=16):
        layers = []
        layers.append(TransitionDown(self.in_planes, planes * block.expansion, stride, nsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)

    def _make_dec(self, block, planes, blocks, share_planes=8, nsample=16, is_head=False):
        layers = []
        layers.append(TransitionUp(self.in_planes, None if is_head else planes * block.expansion))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)

    def label2one_hot(self, labels, C=10):
        b, n = labels.shape
        labels = torch.unsqueeze(labels, dim=1)
        one_hot = torch.zeros(b, C, n, dtype=torch.long).cuda()         # create black
        target = one_hot.scatter_(1, labels.type(torch.long).data, 1)   # retuqire long type
        return target.type(torch.float32)
    
    def label2one_hot_v2(self, labels, C=10):
        n = labels.size(0)
        labels = torch.unsqueeze(labels, dim=0).unsqueeze(0)
        one_hot = torch.zeros(1, C, n, dtype=torch.long).cuda()         # create black
        target = one_hot.scatter_(1, labels.type(torch.long).data, 1)   # retuqire long type
        return target.type(torch.float32)

    def forward(self, pxo, onehot_label=None, label=None, instance_label=None, param=None, normal_s3dis=None):
        p0, x0, o0 = pxo  # (n, 3), (n, c), (b)
        x0 = p0 if self.c == 3 else torch.cat((p0, x0), 1)
        p1, x1, o1 = self.enc1([p0, x0, o0])
        p2, x2, o2 = self.enc2([p1, x1, o1])
        p3, x3, o3 = self.enc3([p2, x2, o2])
        p4, x4, o4 = self.enc4([p3, x3, o3])
        p5, x5, o5 = self.enc5([p4, x4, o4])
        x5 = self.dec5[1:]([p5, self.dec5[0]([p5, x5, o5]), o5])[1]
        x4 = self.dec4[1:]([p4, self.dec4[0]([p4, x4, o4], [p5, x5, o5]), o4])[1]
        x3 = self.dec3[1:]([p3, self.dec3[0]([p3, x3, o3], [p4, x4, o4]), o3])[1]
        x2 = self.dec2[1:]([p2, self.dec2[0]([p2, x2, o2], [p3, x3, o3]), o2])[1]
        x1 = self.dec1[1:]([p1, self.dec1[0]([p1, x1, o1], [p2, x2, o2]), o1])[1]
        # out = self.cls(x1) # n × classes
        # parameter = self.parameter(x1) # n × 3
        # primitive_embedding = self.embedding(x1)    # n × 32 用于超点均值聚类分割
        # boundary_pred = self.boundary(x1)

        # # 可视化特征热力图
        # for i in range(len(o0)):
        #     if i == 0:
        #         p = p0[:o0[i]]
        #         x = x1[:o0[i]]
        #     else:
        #         p = p0[o0[i-1]:o0[i]]
        #         x = x1[o0[i-1]:o0[i]]
        #     # x = torch.mean(x, dim=1)
        #     x = torch.norm(x, dim=1, p=2)
        #     x = torch.clip(x, 3, 5)
        #     # import matplotlib.pyplot as plt
        #     # fig, ax = plt.subplots()
        #     # ax.bar(range(len(x)), x.cpu().detach().numpy())
        #     # plt.savefig('./feature_heatmap_{}.png'.format(i))
        #     norm_x = (x - x.min()) / (x.max() - x.min())
        #     # from sklearn.manifold import TSNE
        #     # tsne = TSNE(n_components=1)
        #     # embedding = tsne.fit_transform(norm_x.cpu().detach().numpy())
        #     import matplotlib.cm as cm
        #     cmap = cm.get_cmap('jet')
        #     colors = cmap(norm_x.cpu().detach().numpy())
        #     # import cv2
        #     # heatmap = cv2.applyColorMap((norm_x.cpu().detach().numpy() * 255).astype(np.uint8), cv2.COLORMAP_JET)
        #     pcd = o3d.geometry.PointCloud()
        #     pcd.points = o3d.utility.Vector3dVector(p.cpu().detach().numpy())
        #     pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
        #     o3d.io.write_point_cloud('visual/feature_heatmap/{}.ply'.format(i), pcd)

        # number of clusters for FPS
        num_clusters = 40
        n_o, count = [int(o0[0].item() * self.rate)], int(o0[0].item() * self.rate)
        for i in range(1, o0.shape[0]):
            count += int((o0[i].item() - o0[i-1].item()) * self.rate)
            n_o.append(count)
        n_o = torch.cuda.IntTensor(n_o) # 以self.rate倍的比例进行采样

        # calculate idx of superpoints and points
        # cluster_idx = pointops_sp.furthestsampling_offset(p0, o0, num_clusters)
        # cluster_idx: b × m
        cluster_idx = pointops.furthestsampling(p0, o0, n_o)    # m
        # cluster_xyz = pointops_sp.gathering_offset(p0.transpose(0, 1).contiguous(), o0, cluster_idx).transpose(1, 2).contiguous()
        # cluster_xyz: b × m × 3
        cluster_xyz = pointops_sp_v2.gathering(p0.transpose(0, 1).contiguous(), cluster_idx).transpose(0, 1).contiguous()


        # c2p_idx: near clusters to each point
        # (b x m x 3, b x m, n x 3) -> b x n x nc2p, b x n x nc2p
        # nc2p == 6
        # c2p_idx, c2p_idx_abs = pointops_sp.knnquerycluster_offset(6, cluster_xyz, cluster_idx, p0, o0)
        c2p_idx, c2p_idx_abs = pointops_sp_v2.knnquerycluster(6, cluster_xyz, cluster_idx, p0, o0, n_o)
        # c2p_idx: n x 6 与每个点最近的nc2p个超点中心索引(基于n)
        # c2p_idx_abs: n x 6 与每个点最近的nc2p个超点中心索引(基于m)

        # association matrix
        # asso_matrix, sp_nei_cnt, sp_lab = pointops_sp.assomatrixpluslabel_offset(6, c2p_idx, label.int().unsqueeze(-1), cluster_idx.unsqueeze(-1), self.classes, o0)
        asso_matrix, sp_nei_cnt, sp_lab = pointops_sp_v2.assomatrixpluslabel(6, o0, n_o, c2p_idx, label.int().unsqueeze(-1), cluster_idx.unsqueeze(-1), self.classes)
        asso_matrix = asso_matrix.float()
        sp_nei_cnt = sp_nei_cnt.float()
        # asso_matrix: b x m x n 点与超点中心关联矩阵
        # sp_nei_cnt: b x m x 1 每个超点所包含点数
        # sp_lab: b x m x class 每个超点中点label数量

        p_fea = x1  # n × 32
        # p_fea = self.spmlp(x1)
        sp_fea = torch.matmul(asso_matrix, x1) / sp_nei_cnt
        # sp_fea = init_fea(p_fea, asso_matrix, sp_nei_cnt, o0)
        # sp_fea: b × m × 32   initial superpoints features

        # c2p_idx: b x n x 6
        # sp_fea, cluster_xyz = self.learn_SLIC_calc_1(sp_fea, cluster_xyz, p_fea, p0, c2p_idx_abs, c2p_idx, cluster_idx, o0)
        sp_fea, cluster_xyz = self.learn_SLIC_calc_1(sp_fea, cluster_xyz, p_fea, p0, c2p_idx_abs, c2p_idx, cluster_idx, o0, n_o)
        # sp_fea: b x m x c
        infoNCE_loss_1 = infoNCE_loss_p2sp(sp_fea, p_fea, c2p_idx_abs, c2p_idx, instance_label)
            
        # sp_fea, cluster_xyz = self.learn_SLIC_calc_2(sp_fea, cluster_xyz, p_fea, p0, c2p_idx_abs, c2p_idx, cluster_idx, o0)
        sp_fea, cluster_xyz = self.learn_SLIC_calc_2(sp_fea, cluster_xyz, p_fea, p0, c2p_idx_abs, c2p_idx, cluster_idx, o0, n_o)
        # sp_fea: b x m x c
        infoNCE_loss_2 = infoNCE_loss_p2sp(sp_fea, p_fea, c2p_idx_abs, c2p_idx, instance_label)

        # sp_fea, cluster_xyz = self.learn_SLIC_calc_3(sp_fea, cluster_xyz, p_fea, p0, c2p_idx_abs, c2p_idx, cluster_idx, o0)
        sp_fea, cluster_xyz = self.learn_SLIC_calc_3(sp_fea, cluster_xyz, p_fea, p0, c2p_idx_abs, c2p_idx, cluster_idx, o0, n_o)
        # sp_fea: b x m x c
        infoNCE_loss_3 = infoNCE_loss_p2sp(sp_fea, p_fea, c2p_idx_abs, c2p_idx, instance_label)

        # fea_dist, sp_fea, cluster_xyz = self.learn_SLIC_calc_4(sp_fea, cluster_xyz, p_fea, p0, c2p_idx_abs, c2p_idx, cluster_idx, o0)
        fea_dist, sp_fea, cluster_xyz, sp_xyz_idx = self.learn_SLIC_calc_4(sp_fea, cluster_xyz, p_fea, p0, c2p_idx_abs, c2p_idx, cluster_idx, o0, n_o)
        # sp_fea: b x m x c
        # fea_dist: n * 6
        # cluster_xyz: b x m x 3
        # sp_xyz_idx: m * 3
        infoNCE_loss_4 = infoNCE_loss_p2sp(sp_fea, p_fea, c2p_idx_abs, c2p_idx, instance_label)
        
        final_asso = fea_dist

        if onehot_label is not None:
            # ------------------------------ reconstruct xyz ----------------------------
            sp_xyz = cluster_xyz
            # sp_xyz: b x m x 3

            # # 可视化超点中心
            # for i in range(len(n_o)):
            #     if i == 0:
            #         sp_coord = sp_xyz[:n_o[i]]
            #     else:
            #         sp_coord = sp_xyz[n_o[i-1]:n_o[i]]
            #     sp_coord = sp_coord.cpu().detach().numpy()
            #     pcd = o3d.geometry.PointCloud()
            #     pcd.points = o3d.utility.Vector3dVector(sp_coord)
            #     o3d.io.write_point_cloud('visual/sp_center_vis/sp_coord_{}.ply'.format(i), pcd)

            p2sp_idx = torch.argmax(final_asso, dim=1, keepdim=False)
            # p2sp_idx: b x n

            # (b x 3 x m,  b x n, b x n x 6) -> (b x 3 x n)
            # re_p_xyz = pointops_sp.gathering_cluster_offset(sp_xyz.transpose(1, 2).contiguous(), p2sp_idx.int(), c2p_idx_abs, o0)
            re_p_xyz = pointops_sp_v2.gathering_cluster(sp_xyz.transpose(0, 1).contiguous(), p2sp_idx.int(), c2p_idx_abs)
            # re_p_xyz: b x 3 x n 每个点最近的超点中心坐标，用于compact loss
            # (b, c, n), idx : (b, m) tensor, idx_3d: (b, m, k)

            # ------------------------------ reconstruct label ----------------------------
            # onehot_label: b x classes x n
            # sp_label = calc_sp_fea(final_asso, onehot_label.transpose(1, 2).contiguous(), 6, c2p_idx, cluster_idx, o0)
            sp_label = calc_sp_fea_v2(final_asso, onehot_label.squeeze(0).transpose(0, 1).contiguous(), 6, c2p_idx, cluster_idx, o0, n_o)
            # sp_label: b x m x classes
            
            # sp_pseudo_lab = torch.argmax(sp_lab, dim=2, keepdim=False)  # b x m
            sp_pseudo_lab = torch.argmax(sp_lab, dim=1, keepdim=False)  # b x m
            # sp_pseudo_lab_onehot = self.label2one_hot(sp_pseudo_lab, self.classes)    # b x class x m
            sp_pseudo_lab_onehot = self.label2one_hot_v2(sp_pseudo_lab, self.classes)    # b x class x m
            # c2p_idx: b x n x 6
            # final_asso: b x n x 6
            # f: b x n x m
            # (b, n, m) X (b, m, classes) -> (b, n, classes)
            # re_p_label = torch.matmul(f, sp_label)
            # re_p_label: b x n x classes
            
            # (b, classes, m), (b, n, 6) -> b x classes x n x 6
            # c2p_label = pointops_sp.grouping_offset(sp_label.transpose(1, 2).contiguous(), c2p_idx_abs, o0)
            c2p_label = pointops_sp_v2.grouping(sp_label.transpose(0, 1).contiguous(), c2p_idx_abs)
            # (b, classes, m), (b, n, 6) -> b x classes x n x 6
            
            # re_p_label = torch.sum(c2p_label * final_asso.unsqueeze(0).unsqueeze(1), dim=-1, keepdim=False)
            re_p_label = torch.sum(c2p_label * final_asso.unsqueeze(0), dim=-1, keepdim=False)
            # re_p_label: b x classes x n

            # ------------------------------ reconstruct normal ----------------------------
            if normal_s3dis is not None:
                normal = normal_s3dis   # s3dis数据集中的法向量
            else:
                normal = pxo[1]
            # sp_normal = calc_sp_fea(final_asso, normal.unsqueeze(0), 6, c2p_idx, cluster_idx, o0)
            sp_normal = calc_sp_fea_v2(final_asso, normal, 6, c2p_idx, cluster_idx, o0, n_o)

            # c2p_normal = pointops_sp.grouping_offset(sp_normal.transpose(1, 2).contiguous(), c2p_idx_abs, o0)
            c2p_normal = pointops_sp_v2.grouping(sp_normal.transpose(0, 1).contiguous(), c2p_idx_abs)
            # re_p_normal = torch.sum(c2p_normal * final_asso.unsqueeze(0).unsqueeze(1), dim=-1, keepdim=False)
            re_p_normal = torch.sum(c2p_normal * final_asso.unsqueeze(0), dim=-1, keepdim=False).transpose(0,1).contiguous()

            # normal_loss = point_normal_similarity_loss(normals, p2sp_idx, c2p_idx_abs, o0)

            normal_distance_weight = torch.norm(re_p_xyz.squeeze(0).transpose(0,1).contiguous() - p0, p=2, dim=1)  # 距离越远，权重越小
            # normal_loss = (1 - (1 - distance_weight) * torch.sum(normal * re_p_normal, dim=1, keepdim=False) / (torch.norm(normal, dim=1, keepdim=False) * torch.norm(re_p_normal, dim=1, keepdim=False) + 1e-8)).mean()   # 添加距离权重
            # normal_loss = (1 - torch.sum(normal * re_p_normal, dim=1, keepdim=False) / (torch.norm(normal, dim=1, keepdim=False) * torch.norm(re_p_normal, dim=1, keepdim=False) + 1e-8)).mean()

            sp_center_normal = pointops_sp_v2.gathering_cluster(re_p_normal.transpose(0, 1).contiguous(), p2sp_idx.int(), c2p_idx).transpose(0, 1).contiguous()
            # sp_center_normal: m x 3 距离每个点最近的超点中心的重建法向量
            # normal_consistency_loss = (1 - torch.sum(sp_center_normal * re_p_normal, dim=1, keepdim=False) / (torch.norm(sp_center_normal, dim=1, keepdim=False) * torch.norm(re_p_normal, dim=1, keepdim=False) + 1e-8)).mean()
            # normal_consistency_loss = (1 - (1 - distance_weight) * torch.sum(sp_center_normal * re_p_normal, dim=1, keepdim=False) / (torch.norm(sp_center_normal, dim=1, keepdim=False) * torch.norm(re_p_normal, dim=1, keepdim=False) + 1e-8)).mean()    #加距离权重
            # 法线一致性损失，计算每个点的重建法向量与距离最近的超点中心的重建法向量的余弦相似度，余弦相似度越大，损失越小
            # normal_loss += 0.1 * normal_consistency_loss

            # ------------------------------ contrast learning ----------------------------
            # c2p_fea = pointops_sp_v2.grouping(sp_fea.transpose(0, 1).contiguous(), c2p_idx_abs)
            # re_p_fea = torch.sum(c2p_fea * final_asso.unsqueeze(0), dim=-1, keepdim=False).transpose(0,1).contiguous()
            # # sp_center_contrast_loss = Contrastive_InfoNCE_loss_sp(sp_fea, sp_xyz_idx, instance_label, n_o)
            # sp_center_contrast_loss = Contrastive_InfoNCE_loss_re_p_fea(re_p_fea, p2sp_idx, c2p_idx_abs, instance_label, o0)
            # sp_center_contrast_loss = torch.tensor(0.0, device=normal_loss.device)  # 暂时不计算sp中心对比损失
            # p2sp_contrast_loss = torch.tensor(0.0, device=normal_loss.device)  # 暂时不计算p2sp对比损失
            contrastive_loss = infoNCE_loss_4  # 每次迭代都计算p2sp对比损失，新的对比损失，点与k个超点进行对比
            # p2sp_contrast_loss = Contrastive_InfoNCE_loss_p2sp(p_fea, p2sp_idx, c2p_idx_abs, sp_fea, sp_xyz_idx, instance_label, n_o)

            # ------------------------------ reconstruct parameters ----------------------------
            sp_param = calc_sp_fea_v2(final_asso, param, 6, c2p_idx, cluster_idx, o0, n_o)
            c2p_param = pointops_sp_v2.grouping(sp_param.transpose(0, 1).contiguous(), c2p_idx_abs)
            re_p_param = torch.sum(c2p_param * final_asso.unsqueeze(0), dim=-1, keepdim=False).transpose(0,1).contiguous()
            # param_loss = torch.mean(torch.norm(re_p_param - param, p=2, dim=1, keepdim=False))

        else:
            re_p_xyz = None
            re_p_label = None

        if normal_s3dis is None:
            type_per_point = self.cls(x1) # n × classes
            embedding = self.embedding64(x1) # n × 128
            # return final_asso, cluster_idx, c2p_idx, c2p_idx_abs, type_per_point, re_p_xyz, re_p_label, fea_dist, p_fea, sp_label.transpose(1, 2).contiguous(), sp_pseudo_lab, sp_pseudo_lab_onehot, normal_loss
            return embedding, final_asso, cluster_idx, c2p_idx_abs, type_per_point, re_p_xyz, re_p_label, p_fea, sp_label, sp_pseudo_lab, sp_pseudo_lab_onehot, re_p_normal, sp_center_normal, normal_distance_weight, re_p_param, contrastive_loss
        else:
            type_per_point = self.cls(x1) # n × classes
            return type_per_point, final_asso, cluster_idx, c2p_idx_abs, re_p_xyz, re_p_label, p_fea, sp_label, sp_pseudo_lab, sp_pseudo_lab_onehot, re_p_normal, sp_center_normal, normal_distance_weight, re_p_param, contrastive_loss
        '''
        final_asso: b*n*6 点与最近6个超点中心关联矩阵, 用于计算评价指标与可视化
        cluster_idx: b*m 超点中心索引(基于n), 用于计算评价指标与可视化
        # c2p_idx: b*n*6 每点最近超点中心索引(基于n)
        c2p_idx_abs: b*n*6 每点最近超点中心索引(基于m), 用于计算评价指标与可视化
        type_per_point: b*classes*n 每点语义类别得分, 用于语义类别loss
        re_p_xyz: b x 3 x n 每个点最近的超点中心坐标, 用于compact loss
        re_p_label: b*classes*n 重建每点的label vector, 用于point label loss
        # fea_dist: b*n*6
        p_fea: b*n*64 每点特征
        sp_label: b*m*13 根据关联矩阵生成超点中心的加权label向量, 用于superpoint label loss
        sp_pseudo_lab: b*m 超点中心伪标签, 投票生成
        sp_pseudo_lab_onehot: b*classes*m 超点中心伪标签,独热向量, 用于superpoint label loss
        re_p_normal: 3*n 重建每点的法向量, 用于normal loss
        sp_center_normal: 3*m 重建每个超点中心的法向量, 用于normal consistency loss
        normal_distance_weight: 距离权重, 用于normal and consistency loss
        re_p_param: 22*n 重建每点的参数, 用于param loss
        contrastive_loss: 对比损失
        '''


def superpoint_seg_repro(**kwargs):
    model = SuperPointNet(PointTransformerBlock, [2, 3, 4, 6, 3], **kwargs)
    return model


# 全卷积超点生成网络
class SuperPointNet_FCN(nn.Module):
    def __init__(self, block, blocks, c=6, k=13, args=None):
        super().__init__()
        self.c = c
        self.classes = k
        self.rate = args.rate
        self.nc2p = args.near_clusters2point
        self.add_rate = args.add_rate
        self.IA_FPS = args.IA_FPS
        self.in_planes, planes = c, [32, 64, 128, 256, 512]
        fpn_planes, fpnhead_planes, share_planes = 128, 64, 8
        stride, nsample = [1, 4, 4, 4, 4], [8, 16, 16, 16, 16]
        self.enc1 = self._make_enc(block, planes[0], blocks[0], share_planes, stride=stride[0], nsample=nsample[0])  # N/1
        self.enc2 = self._make_enc(block, planes[1], blocks[1], share_planes, stride=stride[1], nsample=nsample[1])  # N/4
        self.enc3 = self._make_enc(block, planes[2], blocks[2], share_planes, stride=stride[2], nsample=nsample[2])  # N/16
        self.enc4 = self._make_enc(block, planes[3], blocks[3], share_planes, stride=stride[3], nsample=nsample[3])  # N/64
        self.enc5 = self._make_enc(block, planes[4], blocks[4], share_planes, stride=stride[4], nsample=nsample[4])  # N/256
        self.dec5 = self._make_dec(block, planes[4], 2, share_planes, nsample=nsample[4], is_head=True)  # transform p5
        self.dec4 = self._make_dec(block, planes[3], 2, share_planes, nsample=nsample[3])  # fusion p5 and p4
        self.dec3 = self._make_dec(block, planes[2], 2, share_planes, nsample=nsample[2])  # fusion p4 and p3
        self.dec2 = self._make_dec(block, planes[1], 2, share_planes, nsample=nsample[1])  # fusion p3 and p2
        self.dec1 = self._make_dec(block, planes[0], 2, share_planes, nsample=nsample[0])  # fusion p2 and p1
        self.cls = nn.Sequential(nn.Linear(planes[0], planes[0]), nn.BatchNorm1d(planes[0]), nn.ReLU(inplace=True), nn.Linear(planes[0], k))
        # self.boundary = nn.Sequential(nn.Linear(planes[0], planes[0]), nn.BatchNorm1d(planes[0]), nn.ReLU(inplace=True), nn.Linear(planes[0], 2))
        # self.embedding = nn.Sequential(nn.Linear(planes[0], planes[0]), nn.BatchNorm1d(planes[0]), nn.ReLU(inplace=True), nn.Linear(planes[0], planes[0]))
        # self.embedding64 = nn.Sequential(nn.Linear(planes[0], planes[0]), nn.BatchNorm1d(planes[0]), nn.ReLU(inplace=True), nn.Linear(planes[0], 64))
        # # self.spmlp = nn.Sequential(nn.Linear(planes[0], planes[0]), nn.BatchNorm1d(planes[0]), nn.ReLU(inplace=True), nn.Linear(planes[0], planes[0]))
        # self.parameter = nn.Sequential(nn.Linear(planes[0], planes[0]), nn.BatchNorm1d(planes[0]), nn.ReLU(inplace=True), nn.Linear(planes[0], 22))
        # self.asso = nn.Linear(planes[0], self.nc2p)
        # self.softmax = nn.Softmax(1)

        # self.learn_SLIC_calc_1 = learn_SLIC_calc_v1_new(ch_wc2p_fea=[32, 16, 16], ch_wc2p_xyz=[3, 16, 16], ch_mlp=[32, 16, 16],
        #                     bn=True, use_xyz=True, use_softmax=True, use_norm=False)

        # self.learn_SLIC_calc_2 = learn_SLIC_calc_v1_new(ch_wc2p_fea=[32, 16, 16], ch_wc2p_xyz=[3, 16, 16], ch_mlp=[32, 16, 16],
        #                     bn=True, use_xyz=True, use_softmax=True, use_norm=False)

        # self.learn_SLIC_calc_3 = learn_SLIC_calc_v1_new(ch_wc2p_fea=[32, 16, 16], ch_wc2p_xyz=[3, 16, 16], ch_mlp=[32, 16, 16],
        #                     bn=True, use_xyz=True, use_softmax=True, use_norm=False)

        # self.learn_SLIC_calc_4 = learn_SLIC_calc_v1_new(ch_wc2p_fea=[32, 16, 16], ch_wc2p_xyz=[3, 16, 16], ch_mlp=[32, 16, 16],
        #                     bn=True, use_xyz=True, use_softmax=True, use_norm=False, last=True)

        # self.learn_SLIC_calc_1 = learn_SLIC_calc_v2(ch_wc2p_fea=[32, 16, 16], ch_wc2p_xyz=[3, 16, 16], ch_mlp=[32, 16, 16],
        #                     bn=True, use_xyz=True, use_softmax=True, use_norm=False)

        # self.learn_SLIC_calc_2 = learn_SLIC_calc_v2(ch_wc2p_fea=[32, 16, 16], ch_wc2p_xyz=[3, 16, 16], ch_mlp=[32, 16, 16],
        #                     bn=True, use_xyz=True, use_softmax=True, use_norm=False)

        # self.learn_SLIC_calc_3 = learn_SLIC_calc_v2(ch_wc2p_fea=[32, 16, 16], ch_wc2p_xyz=[3, 16, 16], ch_mlp=[32, 16, 16],
        #                     bn=True, use_xyz=True, use_softmax=True, use_norm=False)

        self.learn_SLIC_calc_4 = learn_SLIC_calc_v2(ch_wc2p_fea=[32, 16, 16], ch_wc2p_xyz=[3, 16, 16], ch_mlp=[32, 16, 16],
                            bn=True, use_xyz=True, use_softmax=True, use_norm=False, last=True)

        # self.superpoint_transformer = superpoint_transformer_v3(ch_wc2p_fea=[32, 16, 16], ch_wc2p_xyz=[3, 16, 16], ch_mlp=[32, 16, 16],
        #                     bn=True, use_xyz=True, use_softmax=True, use_norm=False, last=True)

        # self.superpoint_transformer = superpoint_transformer(ch_wc2p_fea=[32, 16, 16], ch_wc2p_xyz=[3, 16, 16], ch_mlp=[32, 16, 16],
        #                     bn=True, use_xyz=True, use_softmax=True, use_norm=False, last=True)

        # self.superpoint_transformer1 = superpoint_transformer(ch_wc2p_fea=[32, 16, 16], ch_wc2p_xyz=[3, 16, 16], ch_mlp=[32, 16, 16],
        #                     bn=True, use_xyz=True, use_softmax=True, use_norm=False, last=True)

        # self.superpoint_transformer2 = superpoint_transformer(ch_wc2p_fea=[32, 16, 16], ch_wc2p_xyz=[3, 16, 16], ch_mlp=[32, 16, 16],
        #                     bn=True, use_xyz=True, use_softmax=True, use_norm=False, last=True)

        # self.superpoint_transformer3 = superpoint_transformer(ch_wc2p_fea=[32, 16, 16], ch_wc2p_xyz=[3, 16, 16], ch_mlp=[32, 16, 16],
        #                     bn=True, use_xyz=True, use_softmax=True, use_norm=False, last=True)

        # self.superpoint_transformer = superpoint_transformer_v2(ch_wc2p_fea=[32, 16, 16], ch_wc2p_xyz=[3, 16, 16], ch_mlp=[32, 16, 16],
        #                     bn=True, use_xyz=True, use_softmax=True, use_norm=False, last=True)

    def _make_enc(self, block, planes, blocks, share_planes=8, stride=1, nsample=16):
        layers = []
        layers.append(TransitionDown(self.in_planes, planes * block.expansion, stride, nsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)

    def _make_dec(self, block, planes, blocks, share_planes=8, nsample=16, is_head=False):
        layers = []
        layers.append(TransitionUp(self.in_planes, None if is_head else planes * block.expansion))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)

    def label2one_hot(self, labels, C=10):
        b, n = labels.shape
        labels = torch.unsqueeze(labels, dim=1)
        one_hot = torch.zeros(b, C, n, dtype=torch.long).cuda()         # create black
        target = one_hot.scatter_(1, labels.type(torch.long).data, 1)   # retuqire long type
        return target.type(torch.float32)
    
    def label2one_hot_v2(self, labels, C=10):
        n = labels.size(0)
        labels = torch.unsqueeze(labels, dim=0).unsqueeze(0)
        one_hot = torch.zeros(1, C, n, dtype=torch.long).cuda()         # create black
        target = one_hot.scatter_(1, labels.type(torch.long).data, 1)   # retuqire long type
        return target.type(torch.float32)

    def forward(self, pxo, onehot_label=None, label=None, instance_label=None, param=None, normal_s3dis=None):
        p0, x0, o0 = pxo  # (n, 3), (n, c), (b)
        x0 = p0 if self.c == 3 else torch.cat((p0, x0), 1)
        p1, x1, o1 = self.enc1([p0, x0, o0])
        p2, x2, o2 = self.enc2([p1, x1, o1])
        p3, x3, o3 = self.enc3([p2, x2, o2])
        p4, x4, o4 = self.enc4([p3, x3, o3])
        p5, x5, o5 = self.enc5([p4, x4, o4])
        x5 = self.dec5[1:]([p5, self.dec5[0]([p5, x5, o5]), o5])[1]
        x4 = self.dec4[1:]([p4, self.dec4[0]([p4, x4, o4], [p5, x5, o5]), o4])[1]
        x3 = self.dec3[1:]([p3, self.dec3[0]([p3, x3, o3], [p4, x4, o4]), o3])[1]
        x2 = self.dec2[1:]([p2, self.dec2[0]([p2, x2, o2], [p3, x3, o3]), o2])[1]
        x1 = self.dec1[1:]([p1, self.dec1[0]([p1, x1, o1], [p2, x2, o2]), o1])[1]
        type_per_point = self.cls(x1) # n × classes
        # parameter = self.parameter(x1) # n × 3
        # primitive_embedding = self.embedding(x1)    # n × 32 用于超点均值聚类分割
        # boundary_pred = self.boundary(x1)

        # # 可视化特征热力图
        # for i in range(len(o0)):
        #     if i == 0:
        #         p = p0[:o0[i]]
        #         x = x1[:o0[i]]
        #     else:
        #         p = p0[o0[i-1]:o0[i]]
        #         x = x1[o0[i-1]:o0[i]]
        #     # x = torch.mean(x, dim=1)
        #     x = torch.norm(x, dim=1, p=2)
        #     x = torch.clip(x, 3, 5)
        #     # import matplotlib.pyplot as plt
        #     # fig, ax = plt.subplots()
        #     # ax.bar(range(len(x)), x.cpu().detach().numpy())
        #     # plt.savefig('./feature_heatmap_{}.png'.format(i))
        #     norm_x = (x - x.min()) / (x.max() - x.min())
        #     # from sklearn.manifold import TSNE
        #     # tsne = TSNE(n_components=1)
        #     # embedding = tsne.fit_transform(norm_x.cpu().detach().numpy())
        #     import matplotlib.cm as cm
        #     cmap = cm.get_cmap('jet')
        #     colors = cmap(norm_x.cpu().detach().numpy())
        #     # import cv2
        #     # heatmap = cv2.applyColorMap((norm_x.cpu().detach().numpy() * 255).astype(np.uint8), cv2.COLORMAP_JET)
        #     pcd = o3d.geometry.PointCloud()
        #     pcd.points = o3d.utility.Vector3dVector(p.cpu().detach().numpy())
        #     pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
        #     o3d.io.write_point_cloud('visual/feature_heatmap/{}.ply'.format(i), pcd)
        
        if self.IA_FPS:
        # 实例感知FPS, 仅用于test阶段
            point_semantic_scores = F.softmax(type_per_point, dim=1)
            bigger_condition = torch.any(point_semantic_scores[:, :3] >= 0.3, dim=-1)   # 较大语义块的点
            smaller_condition = ~bigger_condition
            for i in range(o0.shape[0]):
                if i == 0:
                    bigger_c = bigger_condition[:o0[i]]
                    bigger_i = torch.nonzero(bigger_c).view(-1).int()
                    smaller_c = smaller_condition[:o0[i]]
                    smaller_i = torch.nonzero(smaller_c).view(-1).int()
                    pt = p0[:o0[i]]
                    pt_big = pt[bigger_c]
                    pt_small = pt[smaller_c]
                    offset_b = torch.cuda.IntTensor([len(pt_big)])
                    offset_s = torch.cuda.IntTensor([len(pt_small)])
                    sp_offset_s = (offset_s * (self.rate+self.add_rate)).int()
                    sp_offset_b = (int(len(pt) * self.rate) - sp_offset_s).int()
                    if sp_offset_s[0] < 1 or sp_offset_b[0] < 1:
                        pt_i = torch.arange(0, o0[i]).to(pt.device).int()
                        offset_pt = torch.cuda.IntTensor([len(pt)])
                        sp_offset_pt = (offset_pt * self.rate).int()
                        cluster_idx_ = pointops.furthestsampling(pt, offset_pt, sp_offset_pt)
                        cluster_idx = pt_i[cluster_idx_.long()]
                        n_o = [len(cluster_idx)]
                        continue
                    cluster_idx_s = pointops.furthestsampling(pt_small, offset_s, sp_offset_s)
                    cluster_idx_b = pointops.furthestsampling(pt_big, offset_b, sp_offset_b)
                    cluster_idx_s = smaller_i[cluster_idx_s.long()]
                    cluster_idx_b = bigger_i[cluster_idx_b.long()]
                    cluster_idx = torch.cat([cluster_idx_s, cluster_idx_b])
                    n_o = [len(cluster_idx)]
                else:
                    bigger_c = bigger_condition[o0[i-1]:o0[i]]
                    bigger_i = torch.nonzero(bigger_c).view(-1).int() + o0[i-1]
                    smaller_c = smaller_condition[o0[i-1]:o0[i]]
                    smaller_i = torch.nonzero(smaller_c).view(-1).int() + o0[i-1]
                    pt = p0[o0[i-1]:o0[i]]
                    pt_big = pt[bigger_c]
                    pt_small = pt[smaller_c]
                    offset_b = torch.cuda.IntTensor([len(pt_big)])
                    offset_s = torch.cuda.IntTensor([len(pt_small)])
                    sp_offset_s = (offset_s * (self.rate+self.add_rate)).int()
                    sp_offset_b = (int(len(pt) * self.rate) - sp_offset_s).int()
                    if sp_offset_s[0] < 1 or sp_offset_b[0] < 1:
                        pt_i = torch.arange(o0[i-1], o0[i]).to(pt.device).int()
                        offset_pt = torch.cuda.IntTensor([len(pt)])
                        sp_offset_pt = (offset_pt * self.rate).int()
                        cluster_idx_ = pointops.furthestsampling(pt, offset_pt, sp_offset_pt)
                        cluster_idx_ = pt_i[cluster_idx_.long()]
                        cluster_idx = torch.cat([cluster_idx, cluster_idx_])
                        n_o.append(len(cluster_idx))
                        continue
                    cluster_idx_s = pointops.furthestsampling(pt_small, offset_s, sp_offset_s)
                    cluster_idx_b = pointops.furthestsampling(pt_big, offset_b, sp_offset_b)
                    cluster_idx_s = smaller_i[cluster_idx_s.long()]
                    cluster_idx_b = bigger_i[cluster_idx_b.long()]
                    cluster_idx_ = torch.cat([cluster_idx_s, cluster_idx_b])
                    cluster_idx = torch.cat([cluster_idx, cluster_idx_])
                    n_o.append(len(cluster_idx))
            n_o = torch.cuda.IntTensor(n_o)
        
        else:
            # number of clusters for FPS
            num_clusters = 40
            n_o, count = [int(o0[0].item() * self.rate)], int(o0[0].item() * self.rate)
            for i in range(1, o0.shape[0]):
                count += int((o0[i].item() - o0[i-1].item()) * self.rate)
                n_o.append(count)
            n_o = torch.cuda.IntTensor(n_o) # 以self.rate倍的比例进行采样

            # calculate idx of superpoints and points
            # cluster_idx = pointops_sp.furthestsampling_offset(p0, o0, num_clusters)
            # cluster_idx: b × m
            cluster_idx = pointops.furthestsampling(p0, o0, n_o)    # m
            # cluster_xyz = pointops_sp.gathering_offset(p0.transpose(0, 1).contiguous(), o0, cluster_idx).transpose(1, 2).contiguous()
            # cluster_xyz: b × m × 3
        cluster_xyz = pointops_sp_v2.gathering(p0.transpose(0, 1).contiguous(), cluster_idx).transpose(0, 1).contiguous()


        # c2p_idx: near clusters to each point
        # (b x m x 3, b x m, n x 3) -> b x n x nc2p, b x n x nc2p
        # nc2p == 6
        # c2p_idx, c2p_idx_abs = pointops_sp.knnquerycluster_offset(6, cluster_xyz, cluster_idx, p0, o0)
        c2p_idx, c2p_idx_abs = pointops_sp_v2.knnquerycluster(self.nc2p, cluster_xyz, cluster_idx, p0, o0, n_o)
        # c2p_idx: n x 6 与每个点最近的nc2p个超点中心索引(基于n)
        # c2p_idx_abs: n x 6 与每个点最近的nc2p个超点中心索引(基于m)

        # association matrix
        # asso_matrix, sp_nei_cnt, sp_lab = pointops_sp.assomatrixpluslabel_offset(6, c2p_idx, label.int().unsqueeze(-1), cluster_idx.unsqueeze(-1), self.classes, o0)
        asso_matrix, sp_nei_cnt, sp_lab = pointops_sp_v2.assomatrixpluslabel(self.nc2p, o0, n_o, c2p_idx, label.int().unsqueeze(-1), cluster_idx.unsqueeze(-1), self.classes)
        asso_matrix = asso_matrix.float()
        sp_nei_cnt = sp_nei_cnt.float()
        # asso_matrix: b x m x n 点与超点中心关联矩阵
        # sp_nei_cnt: b x m x 1 每个超点所包含点数
        # sp_lab: b x m x class 每个超点中点label数量

        p_fea = x1  # n × 32
        # # p_fea = self.spmlp(x1)
        sp_fea = torch.matmul(asso_matrix, p_fea) / sp_nei_cnt
        # # sp_fea = init_fea(p_fea, asso_matrix, sp_nei_cnt, o0)
        # # sp_fea: b × m × 32   initial superpoints features

        # # c2p_idx: b x n x 6
        # # sp_fea, cluster_xyz = self.learn_SLIC_calc_1(sp_fea, cluster_xyz, p_fea, p0, c2p_idx_abs, c2p_idx, cluster_idx, o0)
        # sp_fea, cluster_xyz1 = self.learn_SLIC_calc_1(sp_fea, cluster_xyz, p_fea, p0, c2p_idx_abs, c2p_idx, cluster_idx, o0, n_o)
        # # sp_fea: b x m x c
        # # infoNCE_loss_1 = infoNCE_loss_p2sp(sp_fea, p_fea, c2p_idx_abs, c2p_idx, instance_label)
            
        # # sp_fea, cluster_xyz = self.learn_SLIC_calc_2(sp_fea, cluster_xyz, p_fea, p0, c2p_idx_abs, c2p_idx, cluster_idx, o0)
        # sp_fea, cluster_xyz1 = self.learn_SLIC_calc_2(sp_fea, cluster_xyz, p_fea, p0, c2p_idx_abs, c2p_idx, cluster_idx, o0, n_o)
        # # sp_fea: b x m x c
        # # infoNCE_loss_2 = infoNCE_loss_p2sp(sp_fea, p_fea, c2p_idx_abs, c2p_idx, instance_label)

        # # sp_fea, cluster_xyz = self.learn_SLIC_calc_3(sp_fea, cluster_xyz, p_fea, p0, c2p_idx_abs, c2p_idx, cluster_idx, o0)
        # sp_fea, cluster_xyz1 = self.learn_SLIC_calc_3(sp_fea, cluster_xyz, p_fea, p0, c2p_idx_abs, c2p_idx, cluster_idx, o0, n_o)
        # # sp_fea: b x m x c
        # # infoNCE_loss_3 = infoNCE_loss_p2sp(sp_fea, p_fea, c2p_idx_abs, c2p_idx, instance_label)

        # fea_dist, sp_fea, cluster_xyz = self.learn_SLIC_calc_4(sp_fea, cluster_xyz, p_fea, p0, c2p_idx_abs, c2p_idx, cluster_idx, o0)
        fea_dist, sp_fea1, cluster_xyz, sp_xyz_idx = self.learn_SLIC_calc_4(sp_fea, cluster_xyz, p_fea, p0, c2p_idx_abs, c2p_idx, cluster_idx, o0, n_o)
        # sp_fea: b x m x c
        # fea_dist: n * 6
        # cluster_xyz: b x m x 3
        # sp_xyz_idx: m * 3

        # 超点transformer
        # fea_dist, sp_fea, cluster_xyz = self.superpoint_transformer(sp_fea, cluster_xyz, p_fea, p0, c2p_idx_abs, c2p_idx, cluster_idx, o0, n_o)
        # fea_dist, sp_fea, cluster_xyz = self.superpoint_transformer1(sp_fea, cluster_xyz, p_fea, p0, c2p_idx_abs, c2p_idx, cluster_idx, o0, n_o)
        # fea_dist, sp_fea, cluster_xyz = self.superpoint_transformer2(sp_fea, cluster_xyz, p_fea, p0, c2p_idx_abs, c2p_idx, cluster_idx, o0, n_o)
        # fea_dist, sp_fea, cluster_xyz = self.superpoint_transformer3(sp_fea, cluster_xyz, p_fea, p0, c2p_idx_abs, c2p_idx, cluster_idx, o0, n_o)

        infoNCE_loss_4 = infoNCE_loss_p2sp(sp_fea, p_fea, c2p_idx_abs, c2p_idx, instance_label)
        
        final_asso = fea_dist

        # if onehot_label is not None:
        #     # ------------------------------ reconstruct xyz ----------------------------
        #     sp_xyz = cluster_xyz
        #     # sp_xyz: b x m x 3

        #     # # 可视化超点中心
        #     # for i in range(len(n_o)):
        #     #     if i == 0:
        #     #         sp_coord = sp_xyz[:n_o[i]]
        #     #     else:
        #     #         sp_coord = sp_xyz[n_o[i-1]:n_o[i]]
        #     #     sp_coord = sp_coord.cpu().detach().numpy()
        #     #     pcd = o3d.geometry.PointCloud()
        #     #     pcd.points = o3d.utility.Vector3dVector(sp_coord)
        #     #     o3d.io.write_point_cloud('visual/sp_center_vis/sp_coord_{}.ply'.format(i), pcd)

        #     p2sp_idx = torch.argmax(final_asso, dim=1, keepdim=False)
        #     # p2sp_idx: b x n

        #     # (b x 3 x m,  b x n, b x n x 6) -> (b x 3 x n)
        #     # re_p_xyz = pointops_sp.gathering_cluster_offset(sp_xyz.transpose(1, 2).contiguous(), p2sp_idx.int(), c2p_idx_abs, o0)
        #     re_p_xyz = pointops_sp_v2.gathering_cluster(sp_xyz.transpose(0, 1).contiguous(), p2sp_idx.int(), c2p_idx_abs)
        #     # re_p_xyz: b x 3 x n 每个点最近的超点中心坐标，用于compact loss
        #     # (b, c, n), idx : (b, m) tensor, idx_3d: (b, m, k)

        #     # ------------------------------ reconstruct label ----------------------------
        #     # onehot_label: b x classes x n
        #     # sp_label = calc_sp_fea(final_asso, onehot_label.transpose(1, 2).contiguous(), 6, c2p_idx, cluster_idx, o0)
        #     sp_label = calc_sp_fea_v2(final_asso, onehot_label.squeeze(0).transpose(0, 1).contiguous(), 6, c2p_idx, cluster_idx, o0, n_o)
        #     # sp_label: b x m x classes
            
        #     # sp_pseudo_lab = torch.argmax(sp_lab, dim=2, keepdim=False)  # b x m
        #     sp_pseudo_lab = torch.argmax(sp_lab, dim=1, keepdim=False)  # b x m
        #     # sp_pseudo_lab_onehot = self.label2one_hot(sp_pseudo_lab, self.classes)    # b x class x m
        #     sp_pseudo_lab_onehot = self.label2one_hot_v2(sp_pseudo_lab, self.classes)    # b x class x m
        #     # c2p_idx: b x n x 6
        #     # final_asso: b x n x 6
        #     # f: b x n x m
        #     # (b, n, m) X (b, m, classes) -> (b, n, classes)
        #     # re_p_label = torch.matmul(f, sp_label)
        #     # re_p_label: b x n x classes
            
        #     # (b, classes, m), (b, n, 6) -> b x classes x n x 6
        #     # c2p_label = pointops_sp.grouping_offset(sp_label.transpose(1, 2).contiguous(), c2p_idx_abs, o0)
        #     c2p_label = pointops_sp_v2.grouping(sp_label.transpose(0, 1).contiguous(), c2p_idx_abs)
        #     # (b, classes, m), (b, n, 6) -> b x classes x n x 6
            
        #     # re_p_label = torch.sum(c2p_label * final_asso.unsqueeze(0).unsqueeze(1), dim=-1, keepdim=False)
        #     re_p_label = torch.sum(c2p_label * final_asso.unsqueeze(0), dim=-1, keepdim=False)
        #     # re_p_label: b x classes x n

        #     # ------------------------------ reconstruct normal ----------------------------
        #     if normal_s3dis is not None:
        #         normal = normal_s3dis   # s3dis数据集中的法向量
        #     else:
        #         normal = pxo[1]
        #     # sp_normal = calc_sp_fea(final_asso, normal.unsqueeze(0), 6, c2p_idx, cluster_idx, o0)
        #     sp_normal = calc_sp_fea_v2(final_asso, normal, 6, c2p_idx, cluster_idx, o0, n_o)

        #     # c2p_normal = pointops_sp.grouping_offset(sp_normal.transpose(1, 2).contiguous(), c2p_idx_abs, o0)
        #     c2p_normal = pointops_sp_v2.grouping(sp_normal.transpose(0, 1).contiguous(), c2p_idx_abs)
        #     # re_p_normal = torch.sum(c2p_normal * final_asso.unsqueeze(0).unsqueeze(1), dim=-1, keepdim=False)
        #     re_p_normal = torch.sum(c2p_normal * final_asso.unsqueeze(0), dim=-1, keepdim=False).transpose(0,1).contiguous()

        #     # normal_loss = point_normal_similarity_loss(normals, p2sp_idx, c2p_idx_abs, o0)

        #     normal_distance_weight = torch.norm(re_p_xyz.squeeze(0).transpose(0,1).contiguous() - p0, p=2, dim=1)  # 距离越远，权重越小
        #     # normal_loss = (1 - (1 - distance_weight) * torch.sum(normal * re_p_normal, dim=1, keepdim=False) / (torch.norm(normal, dim=1, keepdim=False) * torch.norm(re_p_normal, dim=1, keepdim=False) + 1e-8)).mean()   # 添加距离权重
        #     # normal_loss = (1 - torch.sum(normal * re_p_normal, dim=1, keepdim=False) / (torch.norm(normal, dim=1, keepdim=False) * torch.norm(re_p_normal, dim=1, keepdim=False) + 1e-8)).mean()

        #     sp_center_normal = pointops_sp_v2.gathering_cluster(re_p_normal.transpose(0, 1).contiguous(), p2sp_idx.int(), c2p_idx).transpose(0, 1).contiguous()
        #     # sp_center_normal: m x 3 距离每个点最近的超点中心的重建法向量
        #     # normal_consistency_loss = (1 - torch.sum(sp_center_normal * re_p_normal, dim=1, keepdim=False) / (torch.norm(sp_center_normal, dim=1, keepdim=False) * torch.norm(re_p_normal, dim=1, keepdim=False) + 1e-8)).mean()
        #     # normal_consistency_loss = (1 - (1 - distance_weight) * torch.sum(sp_center_normal * re_p_normal, dim=1, keepdim=False) / (torch.norm(sp_center_normal, dim=1, keepdim=False) * torch.norm(re_p_normal, dim=1, keepdim=False) + 1e-8)).mean()    #加距离权重
        #     # 法线一致性损失，计算每个点的重建法向量与距离最近的超点中心的重建法向量的余弦相似度，余弦相似度越大，损失越小
        #     # normal_loss += 0.1 * normal_consistency_loss

        #     # ------------------------------ contrast learning ----------------------------
        #     # c2p_fea = pointops_sp_v2.grouping(sp_fea.transpose(0, 1).contiguous(), c2p_idx_abs)
        #     # re_p_fea = torch.sum(c2p_fea * final_asso.unsqueeze(0), dim=-1, keepdim=False).transpose(0,1).contiguous()
        #     # # sp_center_contrast_loss = Contrastive_InfoNCE_loss_sp(sp_fea, sp_xyz_idx, instance_label, n_o)
        #     # sp_center_contrast_loss = Contrastive_InfoNCE_loss_re_p_fea(re_p_fea, p2sp_idx, c2p_idx_abs, instance_label, o0)
        #     # sp_center_contrast_loss = torch.tensor(0.0, device=normal_loss.device)  # 暂时不计算sp中心对比损失
        #     # p2sp_contrast_loss = torch.tensor(0.0, device=normal_loss.device)  # 暂时不计算p2sp对比损失
        #     contrastive_loss = infoNCE_loss_4  # 每次迭代都计算p2sp对比损失，新的对比损失，点与k个超点进行对比
        #     # p2sp_contrast_loss = Contrastive_InfoNCE_loss_p2sp(p_fea, p2sp_idx, c2p_idx_abs, sp_fea, sp_xyz_idx, instance_label, n_o)

        #     # ------------------------------ reconstruct parameters ----------------------------
        #     sp_param = calc_sp_fea_v2(final_asso, param, 6, c2p_idx, cluster_idx, o0, n_o)
        #     c2p_param = pointops_sp_v2.grouping(sp_param.transpose(0, 1).contiguous(), c2p_idx_abs)
        #     re_p_param = torch.sum(c2p_param * final_asso.unsqueeze(0), dim=-1, keepdim=False).transpose(0,1).contiguous()
        #     # param_loss = torch.mean(torch.norm(re_p_param - param, p=2, dim=1, keepdim=False))

        # else:
        #     re_p_xyz = None
        #     re_p_label = None

        # p_fea = x1
        # final_asso = self.asso(x1)  # N * 6
        # final_asso = self.softmax(final_asso)
        p2sp_idx = torch.argmax(final_asso, dim=1, keepdim=False)
        re_p_xyz = pointops_sp_v2.gathering_cluster(cluster_xyz.transpose(0, 1).contiguous(), p2sp_idx.int(), c2p_idx_abs)
        sp_label = calc_sp_fea_v2(final_asso, onehot_label.squeeze(0).transpose(0, 1).contiguous(), self.nc2p, c2p_idx, cluster_idx, o0, n_o)
        c2p_label = pointops_sp_v2.grouping(sp_label.transpose(0, 1).contiguous(), c2p_idx_abs)
        re_p_label = torch.sum(c2p_label * final_asso.unsqueeze(0), dim=-1, keepdim=False)
        sp_pseudo_lab = torch.argmax(sp_lab, dim=1, keepdim=False)  # b x m
        sp_pseudo_lab_onehot = self.label2one_hot_v2(sp_pseudo_lab, self.classes)    # b x class x m

        if normal_s3dis is not None:
            normal = normal_s3dis   # s3dis数据集中的法向量
        else:
            normal = pxo[1]
        sp_normal = calc_sp_fea_v2(final_asso, normal, self.nc2p, c2p_idx, cluster_idx, o0, n_o)
        c2p_normal = pointops_sp_v2.grouping(sp_normal.transpose(0, 1).contiguous(), c2p_idx_abs)
        re_p_normal = torch.sum(c2p_normal * final_asso.unsqueeze(0), dim=-1, keepdim=False).transpose(0,1).contiguous()
        normal_distance_weight = torch.norm(re_p_xyz.squeeze(0).transpose(0,1).contiguous() - p0, p=2, dim=1)  # 距离越远，权重越小
        sp_center_normal = pointops_sp_v2.gathering_cluster(re_p_normal.transpose(0, 1).contiguous(), p2sp_idx.int(), c2p_idx).transpose(0, 1).contiguous()
        contrastive_loss = infoNCE_loss_4  # 每次迭代都计算p2sp对比损失，新的对比损失，点与k个超点进行对比
        # ------------------------------ reconstruct parameters ----------------------------
        sp_param = calc_sp_fea_v2(final_asso, param, self.nc2p, c2p_idx, cluster_idx, o0, n_o)
        c2p_param = pointops_sp_v2.grouping(sp_param.transpose(0, 1).contiguous(), c2p_idx_abs)
        re_p_param = torch.sum(c2p_param * final_asso.unsqueeze(0), dim=-1, keepdim=False).transpose(0,1).contiguous()

        if normal_s3dis is None:
            type_per_point = self.cls(x1) # n × classes
            embedding = self.embedding64(x1) # n × 128
            # return final_asso, cluster_idx, c2p_idx, c2p_idx_abs, type_per_point, re_p_xyz, re_p_label, fea_dist, p_fea, sp_label.transpose(1, 2).contiguous(), sp_pseudo_lab, sp_pseudo_lab_onehot, normal_loss
            return embedding, final_asso, cluster_idx, c2p_idx_abs, type_per_point, re_p_xyz, re_p_label, p_fea, sp_label, sp_pseudo_lab, sp_pseudo_lab_onehot, re_p_normal, sp_center_normal, normal_distance_weight, re_p_param, contrastive_loss
        else:
            return type_per_point, final_asso, cluster_idx, c2p_idx_abs, re_p_xyz, re_p_label, p_fea, sp_label, sp_pseudo_lab, sp_pseudo_lab_onehot, re_p_normal, sp_center_normal, normal_distance_weight, re_p_param, contrastive_loss
        '''
        final_asso: b*n*6 点与最近6个超点中心关联矩阵, 用于计算评价指标与可视化
        cluster_idx: b*m 超点中心索引(基于n), 用于计算评价指标与可视化
        # c2p_idx: b*n*6 每点最近超点中心索引(基于n)
        c2p_idx_abs: b*n*6 每点最近超点中心索引(基于m), 用于计算评价指标与可视化
        type_per_point: b*classes*n 每点语义类别得分, 用于语义类别loss
        re_p_xyz: b x 3 x n 每个点最近的超点中心坐标, 用于compact loss
        re_p_label: b*classes*n 重建每点的label vector, 用于point label loss
        # fea_dist: b*n*6
        p_fea: b*n*64 每点特征
        sp_label: b*m*13 根据关联矩阵生成超点中心的加权label向量, 用于superpoint label loss
        sp_pseudo_lab: b*m 超点中心伪标签, 投票生成
        sp_pseudo_lab_onehot: b*classes*m 超点中心伪标签,独热向量, 用于superpoint label loss
        re_p_normal: 3*n 重建每点的法向量, 用于normal loss
        sp_center_normal: 3*m 重建每个超点中心的法向量, 用于normal consistency loss
        normal_distance_weight: 距离权重, 用于normal and consistency loss
        re_p_param: 22*n 重建每点的参数, 用于param loss
        contrastive_loss: 对比损失
        '''


def superpoint_fcn_seg_repro(**kwargs):
    model = SuperPointNet_FCN(PointTransformerBlock, [2, 3, 4, 6, 3], **kwargs)
    return model



# pointnet超点生成网络
class SuperPointNet_PointNet(nn.Module):
    def __init__(self, c=6, k=13, args=None):
        super().__init__()
        self.c = c
        self.classes = k
        self.rate = args.rate
        self.nc2p = args.near_clusters2point
        self.add_rate = args.add_rate
        self.IA_FPS = args.IA_FPS

        self.backbone = LPE_stn_recurrent(input_channels=2, args=args)

        self.superpoint_transformer = superpoint_transformer(ch_wc2p_fea=[64, 16, 16], ch_wc2p_xyz=[3, 16, 16], ch_mlp=[64, 16, 16],
                            bn=True, use_xyz=True, use_softmax=True, use_norm=False, last=True)

        self.mlp = nn.Sequential(nn.Linear(64, 64), nn.BatchNorm1d(64), nn.Linear(64, k))

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        fea = pc[..., 3:].contiguous()
        return xyz, fea

    def label2one_hot_v2(self, labels, C=10):
        n = labels.size(0)
        labels = torch.unsqueeze(labels, dim=0).unsqueeze(0)
        one_hot = torch.zeros(1, C, n, dtype=torch.long).cuda()         # create black
        target = one_hot.scatter_(1, labels.type(torch.long).data, 1)   # retuqire long type
        return target.type(torch.float32)

    def forward(self, o0, pointcloud: torch.cuda.FloatTensor, clouds_knn, onehot_label=None, label=None, instance_label=None, param=None, normal_s3dis=None):

        xyz, clouds_global = self._break_up_pc(pointcloud)
        # xyz: bn x 3
        # clouds_global: bn x c 特征维度c==7

        n_o, count = [int(o0[0].item() * self.rate)], int(o0[0].item() * self.rate)
        for i in range(1, o0.shape[0]):
            count += int((o0[i].item() - o0[i-1].item()) * self.rate)
            n_o.append(count)
        n_o = torch.cuda.IntTensor(n_o) # 以self.rate倍的比例进行采样

        # calculate idx of superpoints and points
        # cluster_idx = pointops_sp.furthestsampling_offset(p0, o0, num_clusters)
        # cluster_idx: b × m
        cluster_idx = pointops.furthestsampling(xyz, o0, n_o)    # m
        # cluster_xyz = pointops_sp.gathering_offset(p0.transpose(0, 1).contiguous(), o0, cluster_idx).transpose(1, 2).contiguous()
        # cluster_xyz: b × m × 3
        cluster_xyz = pointops_sp_v2.gathering(xyz.transpose(0, 1).contiguous(), cluster_idx).transpose(0, 1).contiguous()

        c2p_idx, c2p_idx_abs = pointops_sp_v2.knnquerycluster(self.nc2p, cluster_xyz, cluster_idx, xyz, o0, n_o)
        # c2p_idx: n x 6 与每个点最近的nc2p个超点中心索引(基于n)
        # c2p_idx_abs: n x 6 与每个点最近的nc2p个超点中心索引(基于m)

        # association matrix
        # asso_matrix, sp_nei_cnt, sp_lab = pointops_sp.assomatrixpluslabel_offset(6, c2p_idx, label.int().unsqueeze(-1), cluster_idx.unsqueeze(-1), self.classes, o0)
        asso_matrix, sp_nei_cnt, sp_lab = pointops_sp_v2.assomatrixpluslabel(self.nc2p, o0, n_o, c2p_idx, label.int().unsqueeze(-1), cluster_idx.unsqueeze(-1), self.classes)
        asso_matrix = asso_matrix.float()
        sp_nei_cnt = sp_nei_cnt.float()
        # asso_matrix: b x m x n 点与超点中心关联矩阵
        # sp_nei_cnt: b x m x 1 每个超点所包含点数
        # sp_lab: b x m x class 每个超点中点label数量

        # ----------------------- embedding ----------------------------
        
        embedding = self.backbone(clouds_knn, clouds_global)
        # embedding: bn x 64

        type_per_point = self.mlp(embedding)

        p_fea = embedding

        sp_fea = torch.matmul(asso_matrix, p_fea) / sp_nei_cnt
        

        # 超点transformer
        fea_dist, sp_fea, cluster_xyz = self.superpoint_transformer(sp_fea, cluster_xyz, p_fea, xyz, c2p_idx_abs, c2p_idx, cluster_idx, o0, n_o)
        # fea_dist, sp_fea, cluster_xyz = self.superpoint_transformer1(sp_fea, cluster_xyz, p_fea, p0, c2p_idx_abs, c2p_idx, cluster_idx, o0, n_o)
        # fea_dist, sp_fea, cluster_xyz = self.superpoint_transformer2(sp_fea, cluster_xyz, p_fea, p0, c2p_idx_abs, c2p_idx, cluster_idx, o0, n_o)
        # fea_dist, sp_fea, cluster_xyz = self.superpoint_transformer3(sp_fea, cluster_xyz, p_fea, p0, c2p_idx_abs, c2p_idx, cluster_idx, o0, n_o)

        infoNCE_loss_4 = infoNCE_loss_p2sp(sp_fea, p_fea, c2p_idx_abs, c2p_idx, instance_label)
        
        final_asso = fea_dist

        p2sp_idx = torch.argmax(final_asso, dim=1, keepdim=False)
        re_p_xyz = pointops_sp_v2.gathering_cluster(cluster_xyz.transpose(0, 1).contiguous(), p2sp_idx.int(), c2p_idx_abs)
        sp_label = calc_sp_fea_v2(final_asso, onehot_label.squeeze(0).transpose(0, 1).contiguous(), self.nc2p, c2p_idx, cluster_idx, o0, n_o)
        c2p_label = pointops_sp_v2.grouping(sp_label.transpose(0, 1).contiguous(), c2p_idx_abs)
        re_p_label = torch.sum(c2p_label * final_asso.unsqueeze(0), dim=-1, keepdim=False)
        sp_pseudo_lab = torch.argmax(sp_lab, dim=1, keepdim=False)  # b x m
        sp_pseudo_lab_onehot = self.label2one_hot_v2(sp_pseudo_lab, self.classes)    # b x class x m

        if normal_s3dis is not None:
            normal = normal_s3dis   # s3dis数据集中的法向量
        else:
            normal = normal_s3dis
        sp_normal = calc_sp_fea_v2(final_asso, normal, self.nc2p, c2p_idx, cluster_idx, o0, n_o)
        c2p_normal = pointops_sp_v2.grouping(sp_normal.transpose(0, 1).contiguous(), c2p_idx_abs)
        re_p_normal = torch.sum(c2p_normal * final_asso.unsqueeze(0), dim=-1, keepdim=False).transpose(0,1).contiguous()
        normal_distance_weight = torch.norm(re_p_xyz.squeeze(0).transpose(0,1).contiguous() - xyz, p=2, dim=1)  # 距离越远，权重越小
        sp_center_normal = pointops_sp_v2.gathering_cluster(re_p_normal.transpose(0, 1).contiguous(), p2sp_idx.int(), c2p_idx).transpose(0, 1).contiguous()
        contrastive_loss = infoNCE_loss_4  # 每次迭代都计算p2sp对比损失，新的对比损失，点与k个超点进行对比
        # ------------------------------ reconstruct parameters ----------------------------
        sp_param = calc_sp_fea_v2(final_asso, param, self.nc2p, c2p_idx, cluster_idx, o0, n_o)
        c2p_param = pointops_sp_v2.grouping(sp_param.transpose(0, 1).contiguous(), c2p_idx_abs)
        re_p_param = torch.sum(c2p_param * final_asso.unsqueeze(0), dim=-1, keepdim=False).transpose(0,1).contiguous()

        if normal_s3dis is None:
            type_per_point = self.cls(embedding) # n × classes
            embedding = self.embedding64(embedding) # n × 128
            # return final_asso, cluster_idx, c2p_idx, c2p_idx_abs, type_per_point, re_p_xyz, re_p_label, fea_dist, p_fea, sp_label.transpose(1, 2).contiguous(), sp_pseudo_lab, sp_pseudo_lab_onehot, normal_loss
            return embedding, final_asso, cluster_idx, c2p_idx_abs, type_per_point, re_p_xyz, re_p_label, p_fea, sp_label, sp_pseudo_lab, sp_pseudo_lab_onehot, re_p_normal, sp_center_normal, normal_distance_weight, re_p_param, contrastive_loss
        else:
            return type_per_point, final_asso, cluster_idx, c2p_idx_abs, re_p_xyz, re_p_label, p_fea, sp_label, sp_pseudo_lab, sp_pseudo_lab_onehot, re_p_normal, sp_center_normal, normal_distance_weight, re_p_param, contrastive_loss


def Superpoint_queryandgroup(c2p_idx, nsample, xyz, new_xyz, feat, idx, offset, new_offset, use_xyz=True):
    """
    input: xyz: (n, 3), new_xyz: (m, 3), feat: (n, c), idx: (m, nsample), offset: (b), new_offset: (b)
    output: new_feat: (m, c+3, nsample), grouped_idx: (m, nsample), c2p_idx: (n, k)
    """
    assert xyz.is_contiguous() and new_xyz.is_contiguous() and feat.is_contiguous()
    if new_xyz is None:
        new_xyz = xyz
    if idx is None:
        idx, _ = pointops.knnquery(nsample, xyz, new_xyz, offset, new_offset) # (m, nsample)
        idx = torch.cat([idx, c2p_idx], dim=1)  # 每个点添加最近的超点作为knn的一部分
        nsample = nsample + c2p_idx.shape[1]

    n, m, c = xyz.shape[0], new_xyz.shape[0], feat.shape[1]
    grouped_xyz = xyz[idx.view(-1).long(), :].view(m, nsample, 3) # (m, nsample, 3)
    #grouped_xyz = grouping(xyz, idx) # (m, nsample, 3)
    grouped_xyz -= new_xyz.unsqueeze(1) # (m, nsample, 3)
    grouped_feat = feat[idx.view(-1).long(), :].view(m, nsample, c) # (m, nsample, c)
    #grouped_feat = grouping(feat, idx) # (m, nsample, c)

    if use_xyz:
        return torch.cat((grouped_xyz, grouped_feat), -1) # (m, nsample, 3+c)
    else:
        return grouped_feat

class PointSuperpointTransformerLayer(nn.Module):
    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        super().__init__()
        self.mid_planes = mid_planes = out_planes // 1
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample
        self.linear_q = nn.Linear(in_planes, mid_planes)
        self.linear_k = nn.Linear(in_planes, mid_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)
        self.linear_p = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3), nn.ReLU(inplace=True), nn.Linear(3, out_planes))
        self.linear_w = nn.Sequential(nn.BatchNorm1d(mid_planes), nn.ReLU(inplace=True),
                                    nn.Linear(mid_planes, mid_planes // share_planes),
                                    nn.BatchNorm1d(mid_planes // share_planes), nn.ReLU(inplace=True),
                                    nn.Linear(out_planes // share_planes, out_planes // share_planes))
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, pxo, c2p_idx) -> torch.Tensor:
        # c2p_idx : 每个点最近的k个超点中心索引（基于n） （n，k）
        p, x, o = pxo  # (n, 3), (n, c), (b)
        x_q, x_k, x_v = self.linear_q(x), self.linear_k(x), self.linear_v(x)  # (n, c)
        x_k = Superpoint_queryandgroup(c2p_idx, self.nsample, p, p, x_k, None, o, o, use_xyz=True)  # (n, nsample, 3+c)
        x_v = Superpoint_queryandgroup(c2p_idx, self.nsample, p, p, x_v, None, o, o, use_xyz=False)  # (n, nsample, c)
        p_r, x_k = x_k[:, :, 0:3], x_k[:, :, 3:]
        for i, layer in enumerate(self.linear_p): p_r = layer(p_r.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i == 1 else layer(p_r)    # (n, nsample, c)
        w = x_k - x_q.unsqueeze(1) + p_r.view(p_r.shape[0], p_r.shape[1], self.out_planes // self.mid_planes, self.mid_planes).sum(2)  # (n, nsample, c)
        for i, layer in enumerate(self.linear_w): w = layer(w.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i % 3 == 0 else layer(w)
        w = self.softmax(w)  # (n, nsample, c)
        n, nsample, c = x_v.shape; s = self.share_planes
        x = ((x_v + p_r).view(n, nsample, s, c // s) * w.unsqueeze(2)).sum(1).view(n, c)
        return x

class PointSuperpointTransformerBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, share_planes=8, nsample=16):
        super(PointSuperpointTransformerBlock, self).__init__()
        self.linear1 = nn.Linear(in_planes, planes, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.transformer2 = PointSuperpointTransformerLayer(planes, planes, share_planes, nsample)
        self.bn2 = nn.BatchNorm1d(planes)
        self.linear3 = nn.Linear(planes, planes * self.expansion, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo, c2p_idx):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        identity = x
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.transformer2([p, x, o], c2p_idx)))
        x = self.bn3(self.linear3(x))
        x += identity
        x = self.relu(x)
        return [p, x, o]

class PointSuperpoingTransformer_Encoder(nn.Module):
    
    def __init__(self, input_channels=6, args=None):
        super(PointSuperpoingTransformer_Encoder, self).__init__()

        self.nsamples = args.get('nsamples', [8, 16, 16, 16, 16])
        self.strides = args.get('strides', [None, 4, 4, 4, 4])
        self.planes = args.get('planes', [64, 128, 256, 512])
        # self.blocks = args.get('blocks', [2, 3, 4, 6, 3])
        self.c = input_channels

        # encoder
        self.in_mlp = nn.Sequential(
            nn.Linear(input_channels, self.planes[0], bias=False),
            nn.BatchNorm1d(self.planes[0]),
            nn.ReLU(inplace=True),
            nn.Linear(self.planes[0], self.planes[0], bias=False),
            nn.BatchNorm1d(self.planes[0]),
            nn.ReLU(inplace=True)
        )

        self.PtBlock1 = PointSuperpointTransformerBlock(self.planes[0], self.planes[0], nsample=10)     # nsample = k + 当前设置值
        self.PtBlock2 = PointSuperpointTransformerBlock(self.planes[0], self.planes[0], nsample=10)
        self.PtBlock3 = PointSuperpointTransformerBlock(self.planes[0], self.planes[0], nsample=10)
        
        self.mlp1 = nn.Linear(256, 1024)
        self.bnmlp1 = nn.BatchNorm1d(1024)

    # def _break_up_pc(self, pc):
    #     xyz = pc[..., 0:3].contiguous().unsqueeze(dim=0)
    #     features = pc[..., 3:].contiguous().unsqueeze(dim=0) if pc.size(-1) > 3 else deepcopy(xyz)
    #     return xyz, features

    def forward(self, pxo, c2p_idx):
        p0, x0, o0 = pxo  # (n, 3), (n, c), (b)
        # x0 = p0 if self.c == 3 else torch.cat((p0, x0), 1)
        
        # encoder
        x0 = self.in_mlp(x0)

        p1, x1, o1 = self.PtBlock1([p0, x0, o0], c2p_idx)
        p2, x2, o2 = self.PtBlock2([p1, x1, o1], c2p_idx)
        p3, x3, o3 = self.PtBlock3([p2, x2, o2], c2p_idx)

        x_features = torch.cat((x0, x1, x2, x3), dim=1)
        x = F.relu(self.bnmlp1(self.mlp1(x_features)))
        # x4 = x.max(dim=2)[0]

        return x, x_features

class PSPTNet(nn.Module):

    def __init__(self, c=6, k=13, args=None):
        super(PSPTNet, self).__init__()

        self.encoder = PointSuperpoingTransformer_Encoder(input_channels=c, args=args)
        self.drop = 0.0

        self.conv1 = torch.nn.Linear(1024 + 256, 512)

        self.bn1 = nn.BatchNorm1d(512)
        self.conv2 = torch.nn.Linear(512, 256)

        self.bn2 = nn.BatchNorm1d(256)

        self.softmax = torch.nn.Softmax(dim=1)
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        self.tanh = torch.nn.Tanh()

        # self.pt_layer = PointTransformerBlock(512)

        self.mlp_embedding_prob1 = torch.nn.Linear(256, 256)
        self.mlp_embedding_prob2 = torch.nn.Linear(256, 32)
        self.bn_embedding_prob1 = nn.BatchNorm1d(256)

        self.mlp_type_prob1 = torch.nn.Linear(256, 256)
        self.mlp_type_prob2 = torch.nn.Linear(256, k)
        self.bn_type_prob1 = nn.BatchNorm1d(256)

    def forward(self, pxo, c2p_idx):

        x, first_layer_features = self.encoder(pxo, c2p_idx)

        x = torch.cat([x, first_layer_features], 1)

        x = F.dropout(F.relu(self.bn1(self.conv1(x))), self.drop)
        x_all = F.dropout(F.relu(self.bn2(self.conv2(x))), self.drop)

        x = F.dropout(F.relu(self.bn_embedding_prob1(self.mlp_embedding_prob1(x_all))), self.drop)
        embedding = self.mlp_embedding_prob2(x)

        x = F.dropout(F.relu(self.bn_type_prob1(self.mlp_type_prob1(x_all))), self.drop)
        type_pred = self.mlp_type_prob2(x)
            
        return embedding, type_pred


class SuperpointNetwork(nn.Module):
    def __init__(self, c=6, k=13, args=None):
        super().__init__()
        self.c = c
        self.classes = k
        self.rate = args.rate
        self.nc2p = args.near_clusters2point
        self.add_rate = args.add_rate
        self.IA_FPS = args.IA_FPS

        self.backbone = PSPTNet(self.c, self.classes, args=args)

        self.superpoint_transformer = superpoint_transformer_v3(ch_wc2p_fea=[32, 16, 16], ch_wc2p_xyz=[3, 16, 16], ch_mlp=[32, 16, 16],
                    bn=True, use_xyz=True, use_softmax=True, use_norm=False, last=True)

        # self.superpoint_transformer = superpoint_transformer_v3(ch_wc2p_fea=[32, 16, 16], ch_wc2p_xyz=[3, 16, 16], ch_mlp=[32, 16, 16],
        #             bn=True, use_xyz=True, use_softmax=True, use_norm=False, last=True)

    def label2one_hot_v2(self, labels, C=10):
        n = labels.size(0)
        labels = torch.unsqueeze(labels, dim=0).unsqueeze(0)
        one_hot = torch.zeros(1, C, n, dtype=torch.long).cuda()         # create black
        target = one_hot.scatter_(1, labels.type(torch.long).data, 1)   # retuqire long type
        return target.type(torch.float32)

    def forward(self, pxo, onehot_label=None, label=None, instance_label=None, param=None, normal_s3dis=None):

        p0, x0, o0 = pxo  # (n, 3), (n, c), (b)
        x0 = p0 if self.c == 3 else torch.cat((p0, x0), 1)

        # number of clusters for FPS
        # num_clusters = 40
        n_o, count = [int(o0[0].item() * self.rate)], int(o0[0].item() * self.rate)
        for i in range(1, o0.shape[0]):
            count += int((o0[i].item() - o0[i-1].item()) * self.rate)
            n_o.append(count)
        n_o = torch.cuda.IntTensor(n_o) # 以self.rate倍的比例进行采样

        # calculate idx of superpoints and points
        # cluster_idx = pointops_sp.furthestsampling_offset(p0, o0, num_clusters)
        # cluster_idx: b × m
        cluster_idx = pointops.furthestsampling(p0, o0, n_o)    # m
        # cluster_xyz = pointops_sp.gathering_offset(p0.transpose(0, 1).contiguous(), o0, cluster_idx).transpose(1, 2).contiguous()
        # cluster_xyz: b × m × 3
        cluster_xyz = pointops_sp_v2.gathering(p0.transpose(0, 1).contiguous(), cluster_idx).transpose(0, 1).contiguous()


        # c2p_idx: near clusters to each point
        # (b x m x 3, b x m, n x 3) -> b x n x nc2p, b x n x nc2p
        # nc2p == 6
        # c2p_idx, c2p_idx_abs = pointops_sp.knnquerycluster_offset(6, cluster_xyz, cluster_idx, p0, o0)
        c2p_idx, c2p_idx_abs = pointops_sp_v2.knnquerycluster(self.nc2p, cluster_xyz, cluster_idx, p0, o0, n_o)
        # c2p_idx: n x 6 与每个点最近的nc2p个超点中心索引(基于n)
        # c2p_idx_abs: n x 6 与每个点最近的nc2p个超点中心索引(基于m)

        embedding, type_per_point = self.backbone([p0, x0, o0], c2p_idx)

        # association matrix
        # asso_matrix, sp_nei_cnt, sp_lab = pointops_sp.assomatrixpluslabel_offset(6, c2p_idx, label.int().unsqueeze(-1), cluster_idx.unsqueeze(-1), self.classes, o0)
        asso_matrix, sp_nei_cnt, sp_lab = pointops_sp_v2.assomatrixpluslabel(self.nc2p, o0, n_o, c2p_idx, label.int().unsqueeze(-1), cluster_idx.unsqueeze(-1), self.classes)
        asso_matrix = asso_matrix.float()
        sp_nei_cnt = sp_nei_cnt.float()
        # asso_matrix: b x m x n 点与超点中心关联矩阵
        # sp_nei_cnt: b x m x 1 每个超点所包含点数
        # sp_lab: b x m x class 每个超点中点label数量

        p_fea = embedding  # n × 64
        # # p_fea = self.spmlp(x1)
        sp_fea = torch.matmul(asso_matrix, p_fea) / sp_nei_cnt
        # # sp_fea = init_fea(p_fea, asso_matrix, sp_nei_cnt, o0)
        # # sp_fea: b × m × 32   initial superpoints features

        # # c2p_idx: b x n x 6
        # # sp_fea, cluster_xyz = self.learn_SLIC_calc_1(sp_fea, cluster_xyz, p_fea, p0, c2p_idx_abs, c2p_idx, cluster_idx, o0)
        # sp_fea, cluster_xyz = self.learn_SLIC_calc_1(sp_fea, cluster_xyz, p_fea, p0, c2p_idx_abs, c2p_idx, cluster_idx, o0, n_o)
        # # sp_fea: b x m x c
        # infoNCE_loss_1 = infoNCE_loss_p2sp(sp_fea, p_fea, c2p_idx_abs, c2p_idx, instance_label)
            
        # # sp_fea, cluster_xyz = self.learn_SLIC_calc_2(sp_fea, cluster_xyz, p_fea, p0, c2p_idx_abs, c2p_idx, cluster_idx, o0)
        # sp_fea, cluster_xyz = self.learn_SLIC_calc_2(sp_fea, cluster_xyz, p_fea, p0, c2p_idx_abs, c2p_idx, cluster_idx, o0, n_o)
        # # sp_fea: b x m x c
        # infoNCE_loss_2 = infoNCE_loss_p2sp(sp_fea, p_fea, c2p_idx_abs, c2p_idx, instance_label)

        # # sp_fea, cluster_xyz = self.learn_SLIC_calc_3(sp_fea, cluster_xyz, p_fea, p0, c2p_idx_abs, c2p_idx, cluster_idx, o0)
        # sp_fea, cluster_xyz = self.learn_SLIC_calc_3(sp_fea, cluster_xyz, p_fea, p0, c2p_idx_abs, c2p_idx, cluster_idx, o0, n_o)
        # # sp_fea: b x m x c
        # infoNCE_loss_3 = infoNCE_loss_p2sp(sp_fea, p_fea, c2p_idx_abs, c2p_idx, instance_label)

        # fea_dist, sp_fea, cluster_xyz = self.learn_SLIC_calc_4(sp_fea, cluster_xyz, p_fea, p0, c2p_idx_abs, c2p_idx, cluster_idx, o0)
        # fea_dist, sp_fea, cluster_xyz, sp_xyz_idx = self.learn_SLIC_calc_4(sp_fea, cluster_xyz, p_fea, p0, c2p_idx_abs, c2p_idx, cluster_idx, o0, n_o)
        # # sp_fea: b x m x c
        # # fea_dist: n * 6
        # # cluster_xyz: b x m x 3
        # # sp_xyz_idx: m * 3

        # 超点transformer
        fea_dist, sp_fea, cluster_xyz = self.superpoint_transformer(sp_fea, cluster_xyz, p_fea, p0, c2p_idx_abs, c2p_idx, cluster_idx, o0, n_o)
        # fea_dist, sp_fea, cluster_xyz = self.superpoint_transformer1(sp_fea, cluster_xyz, p_fea, p0, c2p_idx_abs, c2p_idx, cluster_idx, o0, n_o)
        # fea_dist, sp_fea, cluster_xyz = self.superpoint_transformer2(sp_fea, cluster_xyz, p_fea, p0, c2p_idx_abs, c2p_idx, cluster_idx, o0, n_o)
        # fea_dist, sp_fea, cluster_xyz = self.superpoint_transformer3(sp_fea, cluster_xyz, p_fea, p0, c2p_idx_abs, c2p_idx, cluster_idx, o0, n_o)

        infoNCE_loss_4 = infoNCE_loss_p2sp(sp_fea, p_fea, c2p_idx_abs, c2p_idx, instance_label)
        
        final_asso = fea_dist

        p2sp_idx = torch.argmax(final_asso, dim=1, keepdim=False)
        re_p_xyz = pointops_sp_v2.gathering_cluster(cluster_xyz.transpose(0, 1).contiguous(), p2sp_idx.int(), c2p_idx_abs)
        sp_label = calc_sp_fea_v2(final_asso, onehot_label.squeeze(0).transpose(0, 1).contiguous(), self.nc2p, c2p_idx, cluster_idx, o0, n_o)
        c2p_label = pointops_sp_v2.grouping(sp_label.transpose(0, 1).contiguous(), c2p_idx_abs)
        re_p_label = torch.sum(c2p_label * final_asso.unsqueeze(0), dim=-1, keepdim=False)
        sp_pseudo_lab = torch.argmax(sp_lab, dim=1, keepdim=False)  # b x m
        sp_pseudo_lab_onehot = self.label2one_hot_v2(sp_pseudo_lab, self.classes)    # b x class x m

        if normal_s3dis is not None:
            normal = normal_s3dis   # s3dis数据集中的法向量
        else:
            normal = pxo[1]
        sp_normal = calc_sp_fea_v2(final_asso, normal, self.nc2p, c2p_idx, cluster_idx, o0, n_o)
        c2p_normal = pointops_sp_v2.grouping(sp_normal.transpose(0, 1).contiguous(), c2p_idx_abs)
        re_p_normal = torch.sum(c2p_normal * final_asso.unsqueeze(0), dim=-1, keepdim=False).transpose(0,1).contiguous()
        normal_distance_weight = torch.norm(re_p_xyz.squeeze(0).transpose(0,1).contiguous() - p0, p=2, dim=1)  # 距离越远，权重越小
        sp_center_normal = pointops_sp_v2.gathering_cluster(re_p_normal.transpose(0, 1).contiguous(), p2sp_idx.int(), c2p_idx).transpose(0, 1).contiguous()
        contrastive_loss = infoNCE_loss_4  # 每次迭代都计算p2sp对比损失，新的对比损失，点与k个超点进行对比
        # ------------------------------ reconstruct parameters ----------------------------
        sp_param = calc_sp_fea_v2(final_asso, param, self.nc2p, c2p_idx, cluster_idx, o0, n_o)
        c2p_param = pointops_sp_v2.grouping(sp_param.transpose(0, 1).contiguous(), c2p_idx_abs)
        re_p_param = torch.sum(c2p_param * final_asso.unsqueeze(0), dim=-1, keepdim=False).transpose(0,1).contiguous()

        if normal_s3dis is None:
            type_per_point = type_per_point # n × classes
            embedding = self.embedding64(embedding) # n × 128
            # return final_asso, cluster_idx, c2p_idx, c2p_idx_abs, type_per_point, re_p_xyz, re_p_label, fea_dist, p_fea, sp_label.transpose(1, 2).contiguous(), sp_pseudo_lab, sp_pseudo_lab_onehot, normal_loss
            return embedding, final_asso, cluster_idx, c2p_idx_abs, type_per_point, re_p_xyz, re_p_label, p_fea, sp_label, sp_pseudo_lab, sp_pseudo_lab_onehot, re_p_normal, sp_center_normal, normal_distance_weight, re_p_param, contrastive_loss
        else:
            return type_per_point, final_asso, cluster_idx, c2p_idx_abs, re_p_xyz, re_p_label, p_fea, sp_label, sp_pseudo_lab, sp_pseudo_lab_onehot, re_p_normal, sp_center_normal, normal_distance_weight, re_p_param, contrastive_loss
        '''
        final_asso: b*n*6 点与最近6个超点中心关联矩阵, 用于计算评价指标与可视化
        cluster_idx: b*m 超点中心索引(基于n), 用于计算评价指标与可视化
        # c2p_idx: b*n*6 每点最近超点中心索引(基于n)
        c2p_idx_abs: b*n*6 每点最近超点中心索引(基于m), 用于计算评价指标与可视化
        type_per_point: b*classes*n 每点语义类别得分, 用于语义类别loss
        re_p_xyz: b x 3 x n 每个点最近的超点中心坐标, 用于compact loss
        re_p_label: b*classes*n 重建每点的label vector, 用于point label loss
        # fea_dist: b*n*6
        p_fea: b*n*64 每点特征
        sp_label: b*m*13 根据关联矩阵生成超点中心的加权label向量, 用于superpoint label loss
        sp_pseudo_lab: b*m 超点中心伪标签, 投票生成
        sp_pseudo_lab_onehot: b*classes*m 超点中心伪标签,独热向量, 用于superpoint label loss
        re_p_normal: 3*n 重建每点的法向量, 用于normal loss
        sp_center_normal: 3*m 重建每个超点中心的法向量, 用于normal consistency loss
        normal_distance_weight: 距离权重, 用于normal and consistency loss
        re_p_param: 22*n 重建每点的参数, 用于param loss
        contrastive_loss: 对比损失
        '''


# 全卷积超点生成网络
class PT_seg(nn.Module):
    def __init__(self, block, blocks, c=6, k=13, args=None):
        super().__init__()
        self.c = c
        self.classes = k
        self.rate = args.rate
        self.nc2p = args.near_clusters2point
        self.add_rate = args.add_rate
        self.IA_FPS = args.IA_FPS
        self.in_planes, planes = c, [32, 64, 128, 256, 512]
        fpn_planes, fpnhead_planes, share_planes = 128, 64, 8
        stride, nsample = [1, 4, 4, 4, 4], [8, 16, 16, 16, 16]
        self.enc1 = self._make_enc(block, planes[0], blocks[0], share_planes, stride=stride[0], nsample=nsample[0])  # N/1
        self.enc2 = self._make_enc(block, planes[1], blocks[1], share_planes, stride=stride[1], nsample=nsample[1])  # N/4
        self.enc3 = self._make_enc(block, planes[2], blocks[2], share_planes, stride=stride[2], nsample=nsample[2])  # N/16
        self.enc4 = self._make_enc(block, planes[3], blocks[3], share_planes, stride=stride[3], nsample=nsample[3])  # N/64
        self.enc5 = self._make_enc(block, planes[4], blocks[4], share_planes, stride=stride[4], nsample=nsample[4])  # N/256
        self.dec5 = self._make_dec(block, planes[4], 2, share_planes, nsample=nsample[4], is_head=True)  # transform p5
        self.dec4 = self._make_dec(block, planes[3], 2, share_planes, nsample=nsample[3])  # fusion p5 and p4
        self.dec3 = self._make_dec(block, planes[2], 2, share_planes, nsample=nsample[2])  # fusion p4 and p3
        self.dec2 = self._make_dec(block, planes[1], 2, share_planes, nsample=nsample[1])  # fusion p3 and p2
        self.dec1 = self._make_dec(block, planes[0], 2, share_planes, nsample=nsample[0])  # fusion p2 and p1
        self.cls = nn.Sequential(nn.Linear(planes[0], planes[0]), nn.BatchNorm1d(planes[0]), nn.ReLU(inplace=True), nn.Linear(planes[0], k))

        self.superpoint_transformer = superpoint_transformer(ch_wc2p_fea=[32, 16, 16], ch_wc2p_xyz=[3, 16, 16], ch_mlp=[32, 16, 16],
                            bn=True, use_xyz=True, use_softmax=True, use_norm=False, last=True)

        # self.superpoint_transformer1 = superpoint_transformer(ch_wc2p_fea=[32, 16, 16], ch_wc2p_xyz=[3, 16, 16], ch_mlp=[32, 16, 16],
        #                     bn=True, use_xyz=True, use_softmax=True, use_norm=False, last=True)

        # self.superpoint_transformer2 = superpoint_transformer(ch_wc2p_fea=[32, 16, 16], ch_wc2p_xyz=[3, 16, 16], ch_mlp=[32, 16, 16],
        #                     bn=True, use_xyz=True, use_softmax=True, use_norm=False, last=True)

        # self.superpoint_transformer3 = superpoint_transformer(ch_wc2p_fea=[32, 16, 16], ch_wc2p_xyz=[3, 16, 16], ch_mlp=[32, 16, 16],
        #                     bn=True, use_xyz=True, use_softmax=True, use_norm=False, last=True)

        # self.superpoint_transformer = superpoint_transformer_v2(ch_wc2p_fea=[32, 16, 16], ch_wc2p_xyz=[3, 16, 16], ch_mlp=[32, 16, 16],
        #                     bn=True, use_xyz=True, use_softmax=True, use_norm=False, last=True)

    def _make_enc(self, block, planes, blocks, share_planes=8, stride=1, nsample=16):
        layers = []
        layers.append(TransitionDown(self.in_planes, planes * block.expansion, stride, nsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)

    def _make_dec(self, block, planes, blocks, share_planes=8, nsample=16, is_head=False):
        layers = []
        layers.append(TransitionUp(self.in_planes, None if is_head else planes * block.expansion))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)

    def label2one_hot(self, labels, C=10):
        b, n = labels.shape
        labels = torch.unsqueeze(labels, dim=1)
        one_hot = torch.zeros(b, C, n, dtype=torch.long).cuda()         # create black
        target = one_hot.scatter_(1, labels.type(torch.long).data, 1)   # retuqire long type
        return target.type(torch.float32)
    
    def label2one_hot_v2(self, labels, C=10):
        n = labels.size(0)
        labels = torch.unsqueeze(labels, dim=0).unsqueeze(0)
        one_hot = torch.zeros(1, C, n, dtype=torch.long).cuda()         # create black
        target = one_hot.scatter_(1, labels.type(torch.long).data, 1)   # retuqire long type
        return target.type(torch.float32)

    def forward(self, pxo):
        p0, x0, o0 = pxo  # (n, 3), (n, c), (b)
        x0 = p0 if self.c == 3 else torch.cat((p0, x0), 1)
        p1, x1, o1 = self.enc1([p0, x0, o0])
        p2, x2, o2 = self.enc2([p1, x1, o1])
        p3, x3, o3 = self.enc3([p2, x2, o2])
        p4, x4, o4 = self.enc4([p3, x3, o3])
        p5, x5, o5 = self.enc5([p4, x4, o4])
        x5 = self.dec5[1:]([p5, self.dec5[0]([p5, x5, o5]), o5])[1]
        x4 = self.dec4[1:]([p4, self.dec4[0]([p4, x4, o4], [p5, x5, o5]), o4])[1]
        x3 = self.dec3[1:]([p3, self.dec3[0]([p3, x3, o3], [p4, x4, o4]), o3])[1]
        x2 = self.dec2[1:]([p2, self.dec2[0]([p2, x2, o2], [p3, x3, o3]), o2])[1]
        x1 = self.dec1[1:]([p1, self.dec1[0]([p1, x1, o1], [p2, x2, o2]), o1])[1]
        type_per_point = self.cls(x1) # n × classes
      

        return type_per_point


def pt_seg_repro(**kwargs):
    model = PT_seg(PointTransformerBlock, [2, 3, 4, 6, 3], **kwargs)
    return model