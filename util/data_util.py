import math
import numpy as np
import random
import SharedArray as SA

import torch

from util.voxelize import voxelize


def sa_create(name, var):
    x = SA.create(name, var.shape, dtype=var.dtype)
    x[...] = var[...]
    x.flags.writeable = False
    return x


# def collate_fn(batch):
#     coord, feat, label = list(zip(*batch))
#     offset, count = [], 0
#     for item in coord:
#         count += item.shape[0]
#         offset.append(count)
#     return torch.cat(coord), torch.cat(feat), torch.cat(label), torch.IntTensor(offset)

def collate_fn(batch):
    coord, normals, boundary, label, semantic, param, F, edges, filename, edg_source, edg_target, is_transition = list(zip(*batch))
    offset, count = [], 0
    # print("coord:", len(coord))
    for item in coord:
        # print("item shape:",item.shape)
        count += item.shape[0]
        offset.append(count)

    # F_offset, count = [], 0
    # for item in F:
    #     # print("item shape:",item.shape)
    #     count += item.shape[0]
    #     F_offset.append(count)
    return torch.cat(coord), torch.cat(normals), torch.cat(boundary), torch.cat(label), torch.cat(semantic), torch.cat(param), torch.IntTensor(offset), torch.cat(edges), filename,\
              edg_source, edg_target, is_transition

def collate_fn_limit(batch, max_batch_points, logger):
    coord, normals, boundary, label, semantic, param, F = list(zip(*batch))
    offset, count = [], 0
    # print("coord:", len(coord))
    k = 0
    for item in coord:
        # print("item shape:",item.shape)
        count += item.shape[0]
        if count > max_batch_points:
            break
        k += 1
        offset.append(count)

    if logger is not None and k < len(batch):
        s = sum([x.shape[0] for x in coord])
        s_now = sum([x.shape[0] for x in coord[:k]])
        logger.warning("batch_size shortened from {} to {}, points from {} to {}".format(len(batch), k, s, s_now))

    return torch.cat(coord[:k]), torch.cat(normals[:k]), torch.cat(boundary[:k]), torch.cat(label[:k]), torch.cat(semantic[:k]), torch.cat(param[:k]), torch.IntTensor(offset[:k]), torch.cat(F[:k])

def data_prepare(coord, feat, label, split='train', voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False):
    if transform:
        coord, feat, label = transform(coord, feat, label)
    if voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        uniq_idx = voxelize(coord, voxel_size)
        coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
    if voxel_max and label.shape[0] > voxel_max:
        init_idx = np.random.randint(label.shape[0]) if 'train' in split else label.shape[0] // 2
        crop_idx = np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[:voxel_max]
        coord, feat, label = coord[crop_idx], feat[crop_idx], label[crop_idx]
    if shuffle_index:
        shuf_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuf_idx)
        coord, feat, label = coord[shuf_idx], feat[shuf_idx], label[shuf_idx]

    coord_min = np.min(coord, 0)
    coord -= coord_min
    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat) / 255.
    label = torch.LongTensor(label)
    return coord, feat, label

def data_prepare_abc(coord, normals, boundary, label, semantic, param, F, edges, split='train', voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False):
    # #     if transform:
    #     # coord, feat, label = transform(coord, feat, label)
    #     coord, feat = transform(coord, feat)
    # if voxel_size:
    #     coord_min = np.min(coord, 0)
    #     coord -= coord_min
    #     uniq_idx = voxelize(coord, voxel_size)
    #     coord, normals, boundary, label, semantic, param, F = coord[uniq_idx], normals[uniq_idx], boundary[uniq_idx], label[uniq_idx], semantic[uniq_idx], param[uniq_idx], F[uniq_idx]
    # if voxel_max and label.shape[0] > voxel_max:
    #     init_idx = np.random.randint(label.shape[0]) if 'train' in split else label.shape[0] // 2
    #     crop_idx = np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[:voxel_max]
    #     coord, feat, label = coord[crop_idx], feat[crop_idx], label[crop_idx]
    # if shuffle_index:
    #     shuf_idx = np.arange(coord.shape[0])
    #     np.random.shuffle(shuf_idx)
    #     coord, feat, label = coord[shuf_idx], feat[shuf_idx], label[shuf_idx]

    coord_min = np.min(coord, 0)
    coord -= coord_min
    coord = torch.FloatTensor(coord)
    normals = torch.FloatTensor(normals)
    boundary = torch.LongTensor(boundary)
    semantic = torch.LongTensor(semantic)
    param = torch.FloatTensor(param)
    label = torch.LongTensor(label)
    F = torch.LongTensor(F)
    edges = torch.IntTensor(edges)
    return coord, normals, boundary, label, semantic, param, F, edges

def data_prepare_dse_abc(coord, normals, boundary, label, semantic, param, F, edges, dse_edges, split='train', voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False):

    coord_min = np.min(coord, 0)
    coord -= coord_min
    coord = torch.FloatTensor(coord)
    normals = torch.FloatTensor(normals)
    boundary = torch.LongTensor(boundary)
    semantic = torch.LongTensor(semantic)
    param = torch.FloatTensor(param)
    label = torch.LongTensor(label)
    F = torch.LongTensor(F)
    edges = torch.IntTensor(edges)
    dse_edges = torch.IntTensor(dse_edges)
    return coord, normals, boundary, label, semantic, param, F, edges, dse_edges


def data_prepare_abc_v2(coord, normals, boundary, label, semantic, param, F, edges, edg_source, edg_target, is_transition, split='train', voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False):
    # #     if transform:
    #     # coord, feat, label = transform(coord, feat, label)
    #     coord, feat = transform(coord, feat)
    # if voxel_size:
    #     coord_min = np.min(coord, 0)
    #     coord -= coord_min
    #     uniq_idx = voxelize(coord, voxel_size)
    #     coord, normals, boundary, label, semantic, param, F = coord[uniq_idx], normals[uniq_idx], boundary[uniq_idx], label[uniq_idx], semantic[uniq_idx], param[uniq_idx], F[uniq_idx]
    # if voxel_max and label.shape[0] > voxel_max:
    #     init_idx = np.random.randint(label.shape[0]) if 'train' in split else label.shape[0] // 2
    #     crop_idx = np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[:voxel_max]
    #     coord, feat, label = coord[crop_idx], feat[crop_idx], label[crop_idx]
    # if shuffle_index:
    #     shuf_idx = np.arange(coord.shape[0])
    #     np.random.shuffle(shuf_idx)
    #     coord, feat, label = coord[shuf_idx], feat[shuf_idx], label[shuf_idx]

    coord_min = np.min(coord, 0)
    coord -= coord_min
    coord = torch.FloatTensor(coord)
    normals = torch.FloatTensor(normals)
    boundary = torch.LongTensor(boundary)
    semantic = torch.LongTensor(semantic)
    param = torch.FloatTensor(param)
    label = torch.LongTensor(label)
    F = torch.LongTensor(F)
    edges = torch.IntTensor(edges)
    # edg_source = torch.IntTensor(edg_source)
    # edg_target = torch.IntTensor(edg_target)
    is_transition = torch.IntTensor(is_transition)
    return coord, normals, boundary, label, semantic, param, F, edges, edg_source, edg_target, is_transition

def dataAugment(self, xyz, normal, jitter=False, flip=False, rot=False):
    m = np.eye(3)
    if jitter:
        m += np.random.randn(3, 3) * 0.1
    if flip:
        m[0][0] *= np.random.randint(0, 2) * 2 - 1  # flip x randomly
    if rot:
        theta = np.random.rand() * 2 * math.pi
        m = np.matmul(m, [[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])  # rotation
    return np.matmul(xyz, m), np.matmul(normal, m)