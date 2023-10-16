"""Point transformer
Reference: https://github.com/POSTECH-CVLab/point-transformer
Their result: 70.0 mIoU on S3DIS Area 5. 
"""
from functools import partial
import torch
import torch.nn as nn
import logging

from lib.pointops.functions import pointops
from lib.boundaryops.functions import boundaryops


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
        self.linear_p = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3), nn.ReLU(inplace=True),
                                      nn.Linear(3, out_planes))
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
        for i, layer in enumerate(self.linear_p): p_r = layer(p_r.transpose(1, 2).contiguous()).transpose(1,
                                                                                                          2).contiguous() if i == 1 else layer(
            p_r)  # (n, nsample, c)
        w = x_k - x_q.unsqueeze(1) + p_r.view(p_r.shape[0], p_r.shape[1], self.out_planes // self.mid_planes,
                                              self.mid_planes).sum(2)  # (n, nsample, c)
        for i, layer in enumerate(self.linear_w): w = layer(w.transpose(1, 2).contiguous()).transpose(1,
                                                                                                      2).contiguous() if i % 3 == 0 else layer(
            w)
        w = self.softmax(w)  # (n, nsample, c)
        n, nsample, c = x_v.shape
        s = self.share_planes
        x = ((x_v + p_r).view(n, nsample, s, c // s) * w.unsqueeze(2)).sum(1).view(n, c)
        return x


class EdgeConvLayer(nn.Module):
    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        super().__init__()
        self.mid_planes = mid_planes = out_planes // 1
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample
        self.linear_q = nn.Linear(in_planes, mid_planes)
        self.conv = nn.Sequential(nn.Conv2d(in_planes * 2, out_planes, kernel_size=1, bias=False),
                                  nn.BatchNorm2d(out_planes),
                                  nn.LeakyReLU(negative_slope=0.2))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, pxo) -> torch.Tensor:
        p, x, o = pxo  # (n, 3), (n, c), (b)
        x_k = pointops.queryandgroup(self.nsample, p, p, x, None, o, o, use_xyz=True)
        p_r, x_k = x_k[:, :, 0:3], x_k[:, :, 3:]
        x = x.unsqueeze(1).repeat(1, self.nsample, 1)
        feature = torch.cat((x_k - x, x), dim=2).permute(2, 0, 1).contiguous()
        feature = feature.unsqueeze(0)
        feature = self.conv(feature)
        feature = feature.max(dim=-1, keepdim=False)[0]
        feature = feature.squeeze(0).permute(1, 0).contiguous()
        return feature


class TransitionDown(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, nsample=16):
        super().__init__()
        self.stride, self.nsample = stride, nsample
        if stride != 1:
            self.linear = nn.Linear(3 + in_planes, out_planes, bias=False)
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
                count += (o[i].item() - o[i - 1].item()) // self.stride
                n_o.append(count)
            n_o = torch.cuda.IntTensor(n_o)
            # print(n_o.device, p.device)
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
            self.linear1 = nn.Sequential(nn.Linear(2 * in_planes, in_planes), nn.BatchNorm1d(in_planes),
                                         nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(nn.Linear(in_planes, in_planes), nn.ReLU(inplace=True))
        else:
            self.linear1 = nn.Sequential(nn.Linear(out_planes, out_planes), nn.BatchNorm1d(out_planes),
                                         nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(nn.Linear(in_planes, out_planes), nn.BatchNorm1d(out_planes),
                                         nn.ReLU(inplace=True))

    def forward(self, pxo1, pxo2=None):
        if pxo2 is None:
            _, x, o = pxo1  # (n, 3), (n, c), (b)
            x_tmp = []
            for i in range(o.shape[0]):
                if i == 0:
                    s_i, e_i, cnt = 0, o[0], o[0]
                else:
                    s_i, e_i, cnt = o[i - 1], o[i], o[i] - o[i - 1]
                x_b = x[s_i:e_i, :]
                # cat avg pooling
                x_b = torch.cat((x_b, self.linear2(x_b.sum(0, True) / cnt).repeat(cnt, 1)), 1)
                x_tmp.append(x_b)
            x = torch.cat(x_tmp, 0)
            x = self.linear1(x)
        else:
            p1, x1, o1 = pxo1
            p2, x2, o2 = pxo2
            x = self.linear1(x1) + pointops.interpolation(p2, p1, self.linear2(x2), o2, o1)
        return x


class PointTransformerBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, share_planes=8, nsample=16, **kwargs):
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
    
class BoundaryTransformerBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, share_planes=8, nsample=16, **kwargs):
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


class EdgeConvBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, share_planes=8, nsample=16, mid_res=False):
        super(EdgeConvBlock, self).__init__()
        self.linear1 = nn.Linear(in_planes, planes, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.local_aggr = PointNet2EdgeConvLayer(planes, planes, share_planes, nsample)
        # self.bn2 = nn.BatchNorm1d(planes)
        self.linear3 = nn.Linear(planes, planes * self.expansion, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.mid_res = mid_res

    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        if not self.mid_res:
            identity = x
        x = self.relu(self.bn1(self.linear1(x)))
        if self.mid_res:
            identity = x
        x = self.local_aggr([p, x, o])
        x = self.bn3(self.linear3(x))
        x += identity
        x = self.relu(x)
        return [p, x, o]


class PointNet2EdgeConvLayer(nn.Module):
    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        super().__init__()
        self.mid_planes = mid_planes = out_planes // 1
        self.out_planes = out_planes
        self.nsample = nsample
        # self.linear_q = nn.Linear(in_planes, mid_planes)
        self.conv = nn.Sequential(nn.Conv1d(in_planes + 3, out_planes, kernel_size=1, bias=False),
                                  nn.BatchNorm1d(out_planes),
                                  nn.ReLU(inplace=True))

    def forward(self, pxo) -> torch.Tensor:
        p, x, o = pxo  # (n, 3), (n, c), (b)
        x_k = pointops.queryandgroup(self.nsample, p, p, x, None, o, o, use_xyz=True).transpose(1, 2).contiguous()
        feature = self.conv(x_k)
        feature = feature.max(dim=-1, keepdim=False)[0]
        return feature


class PTSeg(nn.Module):
    def __init__(self,
                 block,
                 blocks,    # depth
                 width=32,
                 nsample=[8, 16, 16, 16, 16],
                 in_channels=6,
                 num_classes=13,
                 dec_local_aggr=True,
                 mid_res=False
                 ):
        super().__init__()
        self.c = in_channels
        self.in_planes, planes = in_channels, [width * 2**i for i in range(len(blocks))]
        fpn_planes, fpnhead_planes, share_planes = 128, 64, 8
        stride, nsample = [1, 4, 4, 4, 4], nsample

        if isinstance(block, str):
            block = eval(block)
        self.mid_res = mid_res
        self.dec_local_aggr = dec_local_aggr
        self.enc1 = self._make_enc(block, planes[0], blocks[0], share_planes, stride=stride[0],
                                   nsample=nsample[0])  # N/1
        self.enc2 = self._make_enc(block, planes[1], blocks[1], share_planes, stride=stride[1],
                                   nsample=nsample[1])  # N/4
        self.enc3 = self._make_enc(block, planes[2], blocks[2], share_planes, stride=stride[2],
                                   nsample=nsample[2])  # N/16
        self.enc4 = self._make_enc(block, planes[3], blocks[3], share_planes, stride=stride[3],
                                   nsample=nsample[3])  # N/64
        self.enc5 = self._make_enc(block, planes[4], blocks[4], share_planes, stride=stride[4],
                                   nsample=nsample[4])  # N/256

        self.dec5 = self._make_dec(block, planes[4], 2, share_planes, nsample[4], True)  # transform p5
        self.dec4 = self._make_dec(block, planes[3], 2, share_planes, nsample[3])  # fusion p5 and p4
        self.dec3 = self._make_dec(block, planes[2], 2, share_planes, nsample[2])  # fusion p4 and p3
        self.dec2 = self._make_dec(block, planes[1], 2, share_planes, nsample[1])  # fusion p3 and p2
        self.dec1 = self._make_dec(block, planes[0], 2, share_planes, nsample[0])  # fusion p2 and p1

        self.dec5_prim = self._make_dec_with_boundary(block, planes[4], 2, share_planes, nsample[4], True)  # transform p5
        self.dec4_prim = self._make_dec_with_boundary(block, planes[3], 2, share_planes, nsample[3])  # fusion p5 and p4
        self.dec3_prim = self._make_dec_with_boundary(block, planes[2], 2, share_planes, nsample[2])  # fusion p4 and p3
        self.dec2_prim = self._make_dec_with_boundary(block, planes[1], 2, share_planes, nsample[1])  # fusion p3 and p2
        self.dec1_prim = self._make_dec_with_boundary(block, planes[0], 2, share_planes, nsample[0])  # fusion p2 and p1


        self.decoder_embedandtype = nn.Sequential(nn.Linear(planes[0], planes[0]), nn.BatchNorm1d(planes[0]), nn.ReLU(inplace=True), nn.Linear(planes[0], planes[0]))
        self.decoder_boundary = nn.Sequential(nn.Linear(planes[0], planes[0]), nn.BatchNorm1d(planes[0]), nn.ReLU(inplace=True), nn.Linear(planes[0], planes[0]))

        self.cls = nn.Sequential(nn.Linear(planes[0], planes[0]), nn.BatchNorm1d(planes[0]), nn.ReLU(inplace=True),
                                 nn.Linear(planes[0], num_classes))
        self.boundary = nn.Sequential(nn.Linear(planes[0], planes[0]), nn.BatchNorm1d(planes[0]), nn.ReLU(inplace=True),
                                 nn.Linear(planes[0], 2))
        self.embedding = nn.Sequential(nn.Linear(planes[0], planes[0]), nn.BatchNorm1d(planes[0]), nn.ReLU(inplace=True),
                                 nn.Linear(planes[0], 128))
        self.softmax = nn.Softmax(dim=1)
        

    def _make_enc(self, block, planes, blocks, share_planes=8, stride=1, nsample=16):
        layers = []
        layers.append(TransitionDown(self.in_planes, planes * block.expansion, stride, nsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample, mid_res=self.mid_res))
        return nn.Sequential(*layers)

    def _make_dec(self, block, planes, blocks, share_planes=8, nsample=16, is_head=False):
        layers = []
        layers.append(TransitionUp(self.in_planes, None if is_head else planes * block.expansion))
        self.in_planes = planes * block.expansion

        if self.dec_local_aggr:
            for _ in range(1, blocks):
                layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample, mid_res=self.mid_res))
        return nn.Sequential(*layers)
    
    def _make_dec_with_boundary(self, block, planes, blocks, share_planes=8, nsample=16, is_head=False):
        layers = []
        layers.append(TransitionUp(self.in_planes, None if is_head else planes * block.expansion))
        self.in_planes = planes * block.expansion
        if planes == 32:
            block = BoundaryTransformerBlock

        if self.dec_local_aggr:
            for _ in range(1, blocks):
                layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample, mid_res=self.mid_res))
        return nn.Sequential(*layers)

    def forward(self, pxo, edges=None):
        # p, x, o: points, features, batches
        # if o0 is None:  # this means p0 is a dict.
        #     p0, x0, o0 = p0['pos'], p0.get('x', None), p0['o']
        # if x0 is None:
        #     x0 = p0
        p0, x0, o0 = pxo  # (n, 3), (n, c), (b)
        x0 = p0 if self.c == 3 else torch.cat((p0, x0), 1)
        p1, x1, o1 = self.enc1([p0, x0, o0])
        p2, x2, o2 = self.enc2([p1, x1, o1])
        p3, x3, o3 = self.enc3([p2, x2, o2])
        p4, x4, o4 = self.enc4([p3, x3, o3])
        p5, x5, o5 = self.enc5([p4, x4, o4])

        # boundary decoder
        x5_b = self.dec5[1:]([p5, self.dec5[0]([p5, x5, o5]), o5])[1]
        x4_b = self.dec4[1:]([p4, self.dec4[0]([p4, x4, o4], [p5, x5_b, o5]), o4])[1]
        x3_b = self.dec3[1:]([p3, self.dec3[0]([p3, x3, o3], [p4, x4_b, o4]), o3])[1]
        x2_b = self.dec2[1:]([p2, self.dec2[0]([p2, x2, o2], [p3, x3_b, o3]), o2])[1]
        x1_b = self.dec1[1:]([p1, self.dec1[0]([p1, x1, o1], [p2, x2_b, o2]), o1])[1]
        boundary_fea = self.decoder_boundary(x1_b)
        boundary = self.boundary(boundary_fea)
        boundary_pred = self.softmax(boundary).clone()
        boundary_pred = (boundary_pred[:, 1] > 0.5).int()

        # primitive decoder
        x5_prim = self.dec5[1:]([p5, self.dec5[0]([p5, x5, o5]), o5])[1]
        x4_prim = self.dec4_prim[1:]([p4, self.dec4[0]([p4, x4, o4], [p5, x5_prim, o5]), o4])[1]
        x3_prim = self.dec3_prim[1:]([p3, self.dec3[0]([p3, x3, o3], [p4, x4_prim, o4]), o3])[1]
        x2_prim = self.dec2_prim[1:]([p2, self.dec2[0]([p2, x2, o2], [p3, x3_prim, o3]), o2])[1]
        x1_prim = self.dec1_prim[1]([p1, self.dec1[0]([p1, x1, o1], [p2, x2_prim, o2]), o1], edges, boundary_pred)[1]
        embedtype_fea = self.decoder_embedandtype(x1_prim)
        embedtype_fea += 0.2*boundary_fea
        type_per_point = self.cls(x1_prim)
        primitive_embedding = self.embedding(embedtype_fea)

        return primitive_embedding, type_per_point, boundary
        # return type_per_point, boundary

def pointtransformer_seg_repro(**kwargs):
    model = PTSeg(PointTransformerBlock, [2, 3, 4, 6, 3], **kwargs)
    return model


class PTSeg128(nn.Module):
    def __init__(self,
                 block,
                 blocks,    # depth
                 width=128,
                 nsample=[16, 16, 16],
                 in_channels=6,
                 num_classes=10,
                 dec_local_aggr=True,
                 mid_res=False
                 ):
        super().__init__()
        self.c = in_channels
        self.in_planes, planes = in_channels, [width * 2**i for i in range(len(blocks))]
        fpn_planes, fpnhead_planes, share_planes = 128, 64, 8
        stride, nsample = [1, 4, 4], nsample

        if isinstance(block, str):
            block = eval(block)
        self.mid_res = mid_res
        self.dec_local_aggr = dec_local_aggr
        self.enc1 = self._make_enc(block, planes[0], blocks[0], share_planes, stride=stride[0],
                                   nsample=nsample[0])  # N/1
        self.enc2 = self._make_enc(block, planes[1], blocks[1], share_planes, stride=stride[1],
                                   nsample=nsample[1])  # N/4
        self.enc3 = self._make_enc(block, planes[2], blocks[2], share_planes, stride=stride[2],
                                   nsample=nsample[2])  # N/16
        # self.enc4 = self._make_enc(block, planes[3], blocks[3], share_planes, stride=stride[3],
        #                            nsample=nsample[3])  # N/64
        # self.enc5 = self._make_enc(block, planes[4], blocks[4], share_planes, stride=stride[4],
        #                            nsample=nsample[4])  # N/256

        # self.dec5 = self._make_dec(block, planes[4], 2, share_planes, nsample[4], True)  # transform p5
        # self.dec4 = self._make_dec(block, planes[3], 2, share_planes, nsample[3])  # fusion p5 and p4
        self.dec3 = self._make_dec(block, planes[2], 2, share_planes, nsample[2], True)  # fusion p4 and p3
        self.dec2 = self._make_dec(block, planes[1], 2, share_planes, nsample[1])  # fusion p3 and p2
        self.dec1 = self._make_dec(block, planes[0], 2, share_planes, nsample[0])  # fusion p2 and p1

        # self.dec5_prim = self._make_dec_with_boundary(block, planes[4], 2, share_planes, nsample[4], True)  # transform p5
        # self.dec4_prim = self._make_dec_with_boundary(block, planes[3], 2, share_planes, nsample[3])  # fusion p5 and p4
        self.dec3_prim = self._make_dec_with_boundary(block, planes[2], 2, share_planes, nsample[2], True)  # fusion p4 and p3
        self.dec2_prim = self._make_dec_with_boundary(block, planes[1], 2, share_planes, nsample[1])  # fusion p3 and p2
        self.dec1_prim = self._make_dec_with_boundary(block, planes[0], 2, share_planes, nsample[0])  # fusion p2 and p1


        self.decoder_embedandtype = nn.Sequential(nn.Linear(planes[0], planes[0]), nn.BatchNorm1d(planes[0]), nn.ReLU(inplace=True), nn.Linear(planes[0], planes[0]))
        self.decoder_boundary = nn.Sequential(nn.Linear(planes[0], planes[0]), nn.BatchNorm1d(planes[0]), nn.ReLU(inplace=True), nn.Linear(planes[0], planes[0]))

        self.cls = nn.Sequential(nn.Linear(planes[0], planes[0]), nn.BatchNorm1d(planes[0]), nn.ReLU(inplace=True),
                                 nn.Linear(planes[0], num_classes))
        self.boundary = nn.Sequential(nn.Linear(planes[0], planes[0]), nn.BatchNorm1d(planes[0]), nn.ReLU(inplace=True),
                                 nn.Linear(planes[0], 2))
        self.embedding = nn.Sequential(nn.Linear(planes[0], planes[0]), nn.BatchNorm1d(planes[0]), nn.ReLU(inplace=True),
                                 nn.Linear(planes[0], 128))
        self.softmax = nn.Softmax(dim=1)
        

    def _make_enc(self, block, planes, blocks, share_planes=8, stride=1, nsample=16):
        layers = []
        layers.append(TransitionDown(self.in_planes, planes * block.expansion, stride, nsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample, mid_res=self.mid_res))
        return nn.Sequential(*layers)

    def _make_dec(self, block, planes, blocks, share_planes=8, nsample=16, is_head=False):
        layers = []
        layers.append(TransitionUp(self.in_planes, None if is_head else planes * block.expansion))
        self.in_planes = planes * block.expansion

        if self.dec_local_aggr:
            for _ in range(1, blocks):
                layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample, mid_res=self.mid_res))
        return nn.Sequential(*layers)
    
    def _make_dec_with_boundary(self, block, planes, blocks, share_planes=8, nsample=16, is_head=False):
        layers = []
        layers.append(TransitionUp(self.in_planes, None if is_head else planes * block.expansion))
        self.in_planes = planes * block.expansion
        if planes == 128:
            block = BoundaryTransformerBlock

        if self.dec_local_aggr:
            for _ in range(1, blocks):
                layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample, mid_res=self.mid_res))
        return nn.Sequential(*layers)

    def forward(self, pxo, edges=None):
        # p, x, o: points, features, batches
        # if o0 is None:  # this means p0 is a dict.
        #     p0, x0, o0 = p0['pos'], p0.get('x', None), p0['o']
        # if x0 is None:
        #     x0 = p0
        p0, x0, o0 = pxo  # (n, 3), (n, c), (b)
        x0 = p0 if self.c == 3 else torch.cat((p0, x0), 1)
        p1, x1, o1 = self.enc1([p0, x0, o0])
        p2, x2, o2 = self.enc2([p1, x1, o1])
        p3, x3, o3 = self.enc3([p2, x2, o2])
        # p4, x4, o4 = self.enc4([p3, x3, o3])
        # p5, x5, o5 = self.enc5([p4, x4, o4])

        # boundary decoder
        # x5_b = self.dec5[1:]([p5, self.dec5[0]([p5, x5, o5]), o5])[1]
        # x4_b = self.dec4[1:]([p4, self.dec4[0]([p4, x4, o4], [p5, x5_b, o5]), o4])[1]
        x3_b = self.dec3[1:]([p3, self.dec3[0]([p3, x3, o3]), o3])[1]
        x2_b = self.dec2[1:]([p2, self.dec2[0]([p2, x2, o2], [p3, x3_b, o3]), o2])[1]
        x1_b = self.dec1[1:]([p1, self.dec1[0]([p1, x1, o1], [p2, x2_b, o2]), o1])[1]
        boundary_fea = self.decoder_boundary(x1_b)
        boundary = self.boundary(boundary_fea)
        boundary_pred = self.softmax(boundary).clone()
        boundary_pred = (boundary_pred[:, 1] > 0.5).int()

        # primitive decoder
        # x5_prim = self.dec5[1:]([p5, self.dec5[0]([p5, x5, o5]), o5])[1]
        # x4_prim = self.dec4_prim[1:]([p4, self.dec4[0]([p4, x4, o4], [p5, x5_prim, o5]), o4])[1]
        x3_prim = self.dec3_prim[1:]([p3, self.dec3[0]([p3, x3, o3]), o3])[1]
        x2_prim = self.dec2_prim[1:]([p2, self.dec2[0]([p2, x2, o2], [p3, x3_prim, o3]), o2])[1]
        x1_prim = self.dec1_prim[1]([p1, self.dec1[0]([p1, x1, o1], [p2, x2_prim, o2]), o1], edges, boundary_pred)[1]
        embedtype_fea = self.decoder_embedandtype(x1_prim)
        embedtype_fea += 0.2*boundary_fea
        type_per_point = self.cls(x1_prim)
        primitive_embedding = self.embedding(embedtype_fea)

        return primitive_embedding, type_per_point, boundary
        # return type_per_point, boundary

def pointtransformer_seg_repro128(**kwargs):
    model = PTSeg128(PointTransformerBlock, [1, 1, 1], **kwargs)
    return model

class PTSeg_RG(nn.Module):
    def __init__(self,
                 block,
                 blocks,    # depth
                 width=32,
                 nsample=[8, 16, 16, 16, 16],
                 in_channels=6,
                 num_classes=13,
                 dec_local_aggr=True,
                 mid_res=False
                 ):
        super().__init__()
        self.c = in_channels
        self.in_planes, planes = in_channels, [width * 2**i for i in range(len(blocks))]
        fpn_planes, fpnhead_planes, share_planes = 128, 64, 8
        stride, nsample = [1, 4, 4, 4, 4], nsample

        if isinstance(block, str):
            block = eval(block)
        self.mid_res = mid_res
        self.dec_local_aggr = dec_local_aggr
        self.enc1 = self._make_enc(block, planes[0], blocks[0], share_planes, stride=stride[0],
                                   nsample=nsample[0])  # N/1
        self.enc2 = self._make_enc(block, planes[1], blocks[1], share_planes, stride=stride[1],
                                   nsample=nsample[1])  # N/4
        self.enc3 = self._make_enc(block, planes[2], blocks[2], share_planes, stride=stride[2],
                                   nsample=nsample[2])  # N/16
        self.enc4 = self._make_enc(block, planes[3], blocks[3], share_planes, stride=stride[3],
                                   nsample=nsample[3])  # N/64
        self.enc5 = self._make_enc(block, planes[4], blocks[4], share_planes, stride=stride[4],
                                   nsample=nsample[4])  # N/256

        self.dec5 = self._make_dec(block, planes[4], 2, share_planes, nsample[4], True)  # transform p5
        self.dec4 = self._make_dec(block, planes[3], 2, share_planes, nsample[3])  # fusion p5 and p4
        self.dec3 = self._make_dec(block, planes[2], 2, share_planes, nsample[2])  # fusion p4 and p3
        self.dec2 = self._make_dec(block, planes[1], 2, share_planes, nsample[1])  # fusion p3 and p2
        self.dec1 = self._make_dec(block, planes[0], 2, share_planes, nsample[0])  # fusion p2 and p1

        self.dec5_prim = self._make_dec_with_boundary(block, planes[4], 2, share_planes, nsample[4], True)  # transform p5
        self.dec4_prim = self._make_dec_with_boundary(block, planes[3], 2, share_planes, nsample[3])  # fusion p5 and p4
        self.dec3_prim = self._make_dec_with_boundary(block, planes[2], 2, share_planes, nsample[2])  # fusion p4 and p3
        self.dec2_prim = self._make_dec_with_boundary(block, planes[1], 2, share_planes, nsample[1])  # fusion p3 and p2
        self.dec1_prim = self._make_dec_with_boundary(block, planes[0], 2, share_planes, nsample[0])  # fusion p2 and p1


        self.decoder_embedandtype = nn.Sequential(nn.Linear(planes[0], planes[0]), nn.BatchNorm1d(planes[0]), nn.ReLU(inplace=True), nn.Linear(planes[0], planes[0]))
        self.decoder_boundary = nn.Sequential(nn.Linear(planes[0], planes[0]), nn.BatchNorm1d(planes[0]), nn.ReLU(inplace=True), nn.Linear(planes[0], planes[0]))

        self.cls = nn.Sequential(nn.Linear(planes[0], planes[0]), nn.BatchNorm1d(planes[0]), nn.ReLU(inplace=True),
                                 nn.Linear(planes[0], num_classes))
        self.boundary = nn.Sequential(nn.Linear(planes[0], planes[0]), nn.BatchNorm1d(planes[0]), nn.ReLU(inplace=True),
                                 nn.Linear(planes[0], 2))
        self.embedding = nn.Sequential(nn.Linear(planes[0], planes[0]), nn.BatchNorm1d(planes[0]), nn.ReLU(inplace=True),
                                 nn.Linear(planes[0], planes[0]))
        self.softmax = nn.Softmax(dim=1)
        

    def _make_enc(self, block, planes, blocks, share_planes=8, stride=1, nsample=16):
        layers = []
        layers.append(TransitionDown(self.in_planes, planes * block.expansion, stride, nsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample, mid_res=self.mid_res))
        return nn.Sequential(*layers)

    def _make_dec(self, block, planes, blocks, share_planes=8, nsample=16, is_head=False):
        layers = []
        layers.append(TransitionUp(self.in_planes, None if is_head else planes * block.expansion))
        self.in_planes = planes * block.expansion

        if self.dec_local_aggr:
            for _ in range(1, blocks):
                layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample, mid_res=self.mid_res))
        return nn.Sequential(*layers)
    
    def _make_dec_with_boundary(self, block, planes, blocks, share_planes=8, nsample=16, is_head=False):
        layers = []
        layers.append(TransitionUp(self.in_planes, None if is_head else planes * block.expansion))
        self.in_planes = planes * block.expansion
        if planes == 32:
            block = BoundaryTransformerBlock

        if self.dec_local_aggr:
            for _ in range(1, blocks):
                layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample, mid_res=self.mid_res))
        return nn.Sequential(*layers)

    def forward(self, pxo, edges=None):
        # p, x, o: points, features, batches
        # if o0 is None:  # this means p0 is a dict.
        #     p0, x0, o0 = p0['pos'], p0.get('x', None), p0['o']
        # if x0 is None:
        #     x0 = p0
        p0, x0, o0 = pxo  # (n, 3), (n, c), (b)
        x0 = p0 if self.c == 3 else torch.cat((p0, x0), 1)
        p1, x1, o1 = self.enc1([p0, x0, o0])
        p2, x2, o2 = self.enc2([p1, x1, o1])
        p3, x3, o3 = self.enc3([p2, x2, o2])
        p4, x4, o4 = self.enc4([p3, x3, o3])
        p5, x5, o5 = self.enc5([p4, x4, o4])

        # boundary decoder
        x5_b = self.dec5[1:]([p5, self.dec5[0]([p5, x5, o5]), o5])[1]
        x4_b = self.dec4[1:]([p4, self.dec4[0]([p4, x4, o4], [p5, x5_b, o5]), o4])[1]
        x3_b = self.dec3[1:]([p3, self.dec3[0]([p3, x3, o3], [p4, x4_b, o4]), o3])[1]
        x2_b = self.dec2[1:]([p2, self.dec2[0]([p2, x2, o2], [p3, x3_b, o3]), o2])[1]
        x1_b = self.dec1[1:]([p1, self.dec1[0]([p1, x1, o1], [p2, x2_b, o2]), o1])[1]
        boundary_fea = self.decoder_boundary(x1_b)
        boundary = self.boundary(boundary_fea)
        # boundary_pred = self.softmax(boundary).clone()
        # boundary_pred = (boundary_pred[:, 1] > 0.5).int()

        # # primitive decoder
        # x5_prim = self.dec5[1:]([p5, self.dec5[0]([p5, x5, o5]), o5])[1]
        # x4_prim = self.dec4_prim[1:]([p4, self.dec4[0]([p4, x4, o4], [p5, x5_prim, o5]), o4])[1]
        # x3_prim = self.dec3_prim[1:]([p3, self.dec3[0]([p3, x3, o3], [p4, x4_prim, o4]), o3])[1]
        # x2_prim = self.dec2_prim[1:]([p2, self.dec2[0]([p2, x2, o2], [p3, x3_prim, o3]), o2])[1]
        # x1_prim = self.dec1_prim[1]([p1, self.dec1[0]([p1, x1, o1], [p2, x2_prim, o2]), o1], edges, boundary_pred)[1]
        # # embedtype_fea = self.decoder_embedandtype(x1_prim)
        # # embedtype_fea += 0.2*boundary_fea
        type_per_point = self.cls(x1_b)
        # primitive_embedding = self.embedding(embedtype_fea)

        # return primitive_embedding, type_per_point, boundary
        return type_per_point, boundary

def pointtransformer_seg_repro_RG(**kwargs):
    model = PTSeg_RG(PointTransformerBlock, [2, 3, 4, 6, 3], **kwargs)
    return model