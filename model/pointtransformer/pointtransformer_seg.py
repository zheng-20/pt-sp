import numpy as np
import queue
import torch
import torch.nn as nn
import trimesh
import torch.nn.functional as F

from lib.pointops.functions import pointops


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
        idx, _ = pointops.knnquery(nsample, xyz, new_xyz, offset, new_offset) # (m, nsample)
        idx = idx.cpu().numpy().tolist()
        idx_1 = []
        boundary = boundary.cpu().numpy()
        for i in range(len(offset)):
            if i == 0:
                xyz_ = xyz[0:offset[i]]
                edges_ = edges[0:offset[i]]
            else:
                xyz_ = xyz[offset[i-1]:offset[i]]
                edges_ = edges[offset[i-1]:offset[i]]
            # edges = trimesh.geometry.faces_to_edges(F_.cpu().numpy())
            # edges = torch.tensor(edges).to(F_.device)

            for j in range(len(xyz_)):
                q = queue.Queue()
                q.put(j)
                n_p = []
                n_p.append(j)

                if boundary[j] == 1:
                    n_p = idx[j]
                else:
                    while(len(n_p) < nsample):
                        if q.qsize() != 0:
                            q_n = q.get()
                        else:
                            n_p = idx[j]
                            break
                        # n, _ = np.where(edges == q_n)
                        # nn_idx = np.unique(edges[n][edges[n] != q_n])
                        # nn_idx = nn_idx[boundary[nn_idx] == 0]
                        nn_idx = edges_[q_n][boundary[edges_[q_n]] == 0]
                        for nn in nn_idx:
                            if nn not in n_p:
                                q.put(nn)
                                n_p.append(nn)
                            if len(n_p) == nsample:
                                break
                # if type(n_p) != torch.Tensor:
                #     n_p = torch.tensor(n_p)
                idx_1.append(n_p)
                # del q
        idx_1 = torch.tensor(idx_1)
        idx = idx_1
                    
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
        x_v, idx = boundary_queryandgroup(self.nsample, p, p, x_v, idx, o, o, edges, boundary, use_xyz=False)  # (n, nsample, c)
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
    def __init__(self, block, blocks, c=6, k=13):
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