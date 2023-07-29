import os
import sys
import glob
import numpy as np
import h5py
import random
from random import choice
import copy
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.init as init
import math
import argparse
from timeit import default_timer as timer
import torchnet as tnt
import functools
import argparse
import transforms3d
from sklearn.linear_model import RANSACRegressor
from plyfile import PlyData, PlyElement

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../'))

# from provider import *
from lib.ply_c import libply_c


def augment_cloud_whole(args, xyz, rgb, normals):
    """" rotate the whole graph, add jitter """
    if args['pc_augm_rot']:
        idx = np.random.randint(xyz.shape[0])
        ref_point = xyz[idx,:3]
        ref_point[2] = 0
        ref_normals = normals[idx,:3]
        ref_normals[2] = 0
        M = transforms3d.axangles.axangle2mat([0,0,1],np.random.uniform(0,2*math.pi)).astype('f4')
        xyz = np.matmul(xyz[:,:3]-ref_point, M)+ref_point
        normals = np.matmul(normals[:,:3]-ref_normals, M)+ref_normals
    if args['pc_augm_jitter']:
        sigma, clip= 0.002, 0.005 # https://github.com/charlesq34/pointnet/blob/master/provider.py#L74
        xyz = xyz + np.clip(sigma * np.random.standard_normal(xyz.shape), -1*clip, clip).astype(np.float32)
        normals = normals + np.clip(sigma * np.random.standard_normal(xyz.shape), -1*clip, clip).astype(np.float32)
        if args['use_rgb']:
            rgb = np.clip(rgb + np.clip(sigma * np.random.standard_normal(xyz.shape), -1*clip, clip).astype(np.float32),-1,1)
    return xyz, rgb, normals

def augment_cloud_whole_v0(args, xyz, rgb):
    """" rotate the whole graph, add jitter """
    if args['pc_augm_rot']:
        ref_point = xyz[np.random.randint(xyz.shape[0]),:3]
        ref_point[2] = 0
        M = transforms3d.axangles.axangle2mat([0,0,1],np.random.uniform(0,2*math.pi)).astype('f4')
        xyz = np.matmul(xyz[:,:3]-ref_point, M)+ref_point
    if args['pc_augm_jitter']:
        sigma, clip= 0.002, 0.005 # https://github.com/charlesq34/pointnet/blob/master/provider.py#L74
        xyz = xyz + np.clip(sigma * np.random.standard_normal(xyz.shape), -1*clip, clip).astype(np.float32)
        if args['use_rgb']:
            rgb = np.clip(rgb + np.clip(sigma * np.random.standard_normal(xyz.shape), -1*clip, clip).astype(np.float32),-1,1)
    return xyz, rgb

def read_structure(file_name, read_geof):
    """
    read the input point cloud in a format ready for embedding    
    """
    data_file = h5py.File(file_name, 'r')

    xyz = np.array(data_file['xyz'], dtype='float32')
    rgb = np.array(data_file['rgb'], dtype='float32')
    elevation = np.array(data_file['elevation'], dtype='float32')
    xyn = np.array(data_file['xyn'], dtype='float32')
    edg_source = np.array(data_file['source'], dtype='int').squeeze()
    edg_target = np.array(data_file['target'], dtype='int').squeeze()
    is_transition = np.array(data_file['is_transition'])
    objects = np.array(data_file['objects'][()])
    labels = np.array(data_file['labels']).squeeze()

    if len(labels.shape) == 0:
        labels = np.array([0])
    if len(is_transition.shape) == 0:
        is_transition = np.array([0])
    if read_geof:
        local_geometry = np.array(data_file['geof'], dtype='float32')
    else:
        local_geometry = np.array(data_file['target_local_geometry'], dtype='uint32')
	
    geof = np.array(data_file['geof'], dtype='float32')

    data_file.close()
    return xyz, rgb, edg_source, edg_target, is_transition, local_geometry, geof, labels, objects, elevation, xyn

def create_vkitti_datasets(args, logger, test_seed_offset=0 ):
    """ Gets training and test datasets. """
    # Load formatted clouds
    testlist, trainlist = [], []
    for n in range(1,7):
        if n != args['cvfold']: 
            path = os.path.join(args['data_root'], '0{:d}/'.format(n))
            for fname in sorted(os.listdir(path), key=lambda x:os.stat(path + "/" + x).st_size):
                if fname.endswith(".h5"):
                    trainlist.append(path+fname)
    
    print('train list: {}'.format(len(trainlist)))
    path = os.path.join(args['data_root'], '0{:d}/'.format(args['cvfold']))
    
    # for fname in sorted(os.listdir(path),key=lambda x:os.stat(path + "/" + x).st_size):
    for fname in sorted(os.listdir(path),key=lambda x:os.stat(path + "/" + x).st_size, reverse=True):
        if fname.endswith(".h5"):
            testlist.append(path+fname)
    # print('test list: {}'.format(testlist))
    print('test list: {}'.format(len(testlist)))
    return tnt.dataset.ListDataset(trainlist,
                                   functools.partial(graph_loader, train=True, args=args, logger=logger, db_path=args['data_root'])),\
            tnt.dataset.ListDataset(testlist,
                                   functools.partial(graph_loader, train=False, args=args, logger=logger, db_path=args['data_root']))


def graph_loader(entry, train, args, logger, db_path, test_seed_offset=0, full_cpu = False):
    """ Load the point cloud and the graph structure """
    xyz, rgb, edg_source, edg_target, is_transition, local_geometry, geof, \
            labels, objects, elevation, xyn = read_structure(entry, False)
    # xyz: 点云坐标 n*3
    # rgb: 点云特征（rgb）n*3
    # edg_source: 点云5邻域种子点索引 5n
    # edg_target: 点云5邻域邻域点索引 5n
    # is_transition: 5邻域点是否是同一个物体，false为1 5n
    # local_geometry: 点云20邻域点索引 n*20
    # geof: 点云几何特征 n*4
    # labels: 语义标注 n*14
    # objects: 所属语义物体索引 n
    # elevation: 使用简单的 平面 模型计算高程 n
    # xyn: 计算 xy 归一化位置 n*2
    
    short_name= entry.split(os.sep)[-2]+'_'+entry.split(os.sep)[-1]
    raw_xyz=np.array(xyz)
    raw_rgb=np.array(rgb)
    raw_labels=np.array(labels)
    rgb = rgb/255

    n_ver = np.shape(xyz)[0]    # n
    n_edg = np.shape(edg_source)[0] # 5n
    selected_ver = np.full((n_ver,), True, dtype='?')
    selected_edg = np.full((n_edg,), True, dtype='?')

    if train:
        xyz, rgb = augment_cloud_whole_v0(args, xyz, rgb)

    subsample = False
    new_ver_index = []

    # if train and (0 < args['num_point'] < n_ver):
    if (0 < args['num_point'] < n_ver):
        subsample = True
        selected_edg, selected_ver = libply_c.random_subgraph(n_ver, 
                                                              edg_source.astype('uint32'),
                                                              edg_target.astype('uint32'),
                                                              int(args['num_point']))
        selected_edg = selected_edg.astype('?')
        selected_ver = selected_ver.astype('?')

        new_ver_index = -np.ones((n_ver,), dtype = int)
        new_ver_index[selected_ver.nonzero()] = range(selected_ver.sum())

        edg_source = new_ver_index[edg_source[selected_edg.astype('?')]]
        edg_target = new_ver_index[edg_target[selected_edg.astype('?')]]

        is_transition = is_transition[selected_edg]
        labels = raw_labels[selected_ver,]
        objects = objects[selected_ver,]
        elevation = elevation[selected_ver]
        xyn = xyn[selected_ver,]

    # if args['learned_embeddings']:
    #     # we use point nets to embed the point clouds
    #     # local_geometry: N x 20  index
    #     nei = local_geometry[selected_ver, :args['k_nn_local']].astype('int64')
        
    #     clouds, clouds_global = [], []
    #     #clouds_global is cloud global features. here, just the diameter + elevation

    #     clouds = xyz[nei,]
    #     #diameters = np.max(np.max(clouds,axis=1) - np.min(clouds,axis=1), axis = 1)
    #     diameters = np.sqrt(clouds.var(1).sum(1))
    #     clouds = (clouds - xyz[selected_ver,np.newaxis,:]) / (diameters[:,np.newaxis,np.newaxis] + 1e-10)

    #     if args['use_rgb']:
    #         clouds = np.concatenate([clouds, rgb[nei,]],axis=2)

    #     if args['ver_value'] == 'geof':                             # N x 4
    #         clouds = np.concatenate([clouds, geof[nei,]],axis=2)    # n x 20 x (xyz+rgb+geof) = n x 20 x 10

    #     clouds = clouds.transpose([0,2,1])

    #     clouds_global = diameters[:,None]
    #     if 'e' in args['global_feat']:
    #         clouds_global = np.hstack((clouds_global, elevation[:,None]))
    #     if 'rgb' in args['global_feat']:
    #         clouds_global = np.hstack((clouds_global, rgb[selected_ver,]))
    #     if 'XY' in args['global_feat']:
    #         clouds_global = np.hstack((clouds_global, xyn))
    #     if 'xy' in args['global_feat']:
    #         clouds_global = np.hstack((clouds_global, xyz[selected_ver,:2]))
    #     if 'o' in args['global_feat']:
    #         clouds_global = np.hstack((clouds_global, geof[selected_ver,]))

    xyz = xyz[selected_ver,]
    rgb = rgb[selected_ver,]

    is_transition = torch.from_numpy(is_transition)
    labels = torch.from_numpy(labels)
    objects = torch.from_numpy(objects.astype('int64'))

    # clouds = torch.from_numpy(clouds)
    # clouds_global = torch.from_numpy(clouds_global)
    # if clouds_global.shape[0] != 15000:
    #     print(short_name)

    xyz = torch.from_numpy(xyz)
    rgb = torch.from_numpy(rgb)

    del raw_labels
    del raw_rgb
    del raw_xyz
    # del nei

    return short_name, edg_source, edg_target, is_transition, labels, objects, xyz, rgb


def collate_vkitti(batch):
    short_name, edg_source, edg_target, is_transition, labels, objects, xyz, rgb = list(zip(*batch))

    offset, count = [], 0
    # print("coord:", len(coord))
    for item in xyz:
        # print("item shape:",item.shape)
        count += item.shape[0]
        offset.append(count)

    xyz = torch.cat(xyz)
    rgb = torch.cat(rgb)
    objects = torch.cat(objects)
    label = torch.cat(labels)
    offset = torch.IntTensor(offset)

    return short_name, xyz, rgb, objects, label, offset, edg_source, edg_target, is_transition


class vkitti_Dataset(Dataset):
    def __init__(self, args, split='train'):
        super().__init__()
        self.split = split
        self.args = args

        """ Gets training and test datasets. """
        # Load formatted clouds
        testlist, trainlist = [], []
        for n in range(1,7):
            if n != args['cvfold']: 
                path = os.path.join(args['data_root'], '0{:d}/'.format(n))
                for fname in sorted(os.listdir(path), key=lambda x:os.stat(path + "/" + x).st_size):
                    if fname.endswith(".h5"):
                        trainlist.append(path+fname)
        
        # print('train list: {}'.format(len(trainlist)))
        path = os.path.join(args['data_root'], '0{:d}/'.format(args['cvfold']))
        
        # for fname in sorted(os.listdir(path),key=lambda x:os.stat(path + "/" + x).st_size):
        for fname in sorted(os.listdir(path),key=lambda x:os.stat(path + "/" + x).st_size, reverse=True):
            if fname.endswith(".h5"):
                testlist.append(path+fname)
        # print('test list: {}'.format(testlist))
        # print('test list: {}'.format(len(testlist)))

        if split == 'train':
            self.data_list = trainlist
        elif split == 'test':
            self.data_list = testlist

        self.data_idx = np.arange(len(self.data_list))
        print("Totally {} samples in {} set.".format(len(self.data_idx), split))

    def __getitem__(self, index):
        data_idx = self.data_idx[index % len(self.data_idx)]
        data = self.data_list[data_idx]

        args = self.args

        """ Load the point cloud and the graph structure """
        xyz, rgb, edg_source, edg_target, is_transition, local_geometry, geof, \
                labels, objects, elevation, xyn = read_structure(data, False)
        # xyz: 点云坐标 n*3
        # rgb: 点云特征（rgb）n*3
        # edg_source: 点云5邻域种子点索引 5n
        # edg_target: 点云5邻域邻域点索引 5n
        # is_transition: 5邻域点是否是同一个物体，false为1 5n
        # local_geometry: 点云20邻域点索引 n*20
        # geof: 点云几何特征 n*4
        # labels: 语义标注 n*14
        # objects: 所属语义物体索引 n
        # elevation: 使用简单的 平面 模型计算高程 n
        # xyn: 计算 xy 归一化位置 n*2
        
        short_name= data.split(os.sep)[-2]+'_'+data.split(os.sep)[-1]
        raw_xyz=np.array(xyz)
        raw_rgb=np.array(rgb)
        raw_labels=np.array(labels)
        rgb = rgb/255

        n_ver = np.shape(xyz)[0]    # n
        n_edg = np.shape(edg_source)[0] # 5n
        selected_ver = np.full((n_ver,), True, dtype='?')
        selected_edg = np.full((n_edg,), True, dtype='?')

        if self.split == 'train':
            xyz, rgb = augment_cloud_whole_v0(args, xyz, rgb)

        subsample = False
        new_ver_index = []

        if self.split == 'train' and (0 < args['num_point'] < n_ver):
        # if (0 < args['num_point'] < n_ver):
            subsample = True
            selected_edg, selected_ver = libply_c.random_subgraph(n_ver, 
                                                                edg_source.astype('uint32'),
                                                                edg_target.astype('uint32'),
                                                                int(args['num_point']))
            selected_edg = selected_edg.astype('?')
            selected_ver = selected_ver.astype('?')

            new_ver_index = -np.ones((n_ver,), dtype = int)
            new_ver_index[selected_ver.nonzero()] = range(selected_ver.sum())

            edg_source = new_ver_index[edg_source[selected_edg.astype('?')]]
            edg_target = new_ver_index[edg_target[selected_edg.astype('?')]]

            is_transition = is_transition[selected_edg]
            labels = raw_labels[selected_ver,]
            objects = objects[selected_ver,]
            # elevation = elevation[selected_ver]
            # xyn = xyn[selected_ver,]

        # if args['learned_embeddings']:
        #     # we use point nets to embed the point clouds
        #     # local_geometry: N x 20  index
        #     nei = local_geometry[selected_ver, :args['k_nn_local']].astype('int64')
            
        #     clouds, clouds_global = [], []
        #     #clouds_global is cloud global features. here, just the diameter + elevation

        #     clouds = xyz[nei,]
        #     #diameters = np.max(np.max(clouds,axis=1) - np.min(clouds,axis=1), axis = 1)
        #     diameters = np.sqrt(clouds.var(1).sum(1))
        #     clouds = (clouds - xyz[selected_ver,np.newaxis,:]) / (diameters[:,np.newaxis,np.newaxis] + 1e-10)

        #     if args['use_rgb']:
        #         clouds = np.concatenate([clouds, rgb[nei,]],axis=2)

        #     if args['ver_value'] == 'geof':                             # N x 4
        #         clouds = np.concatenate([clouds, geof[nei,]],axis=2)    # n x 20 x (xyz+rgb+geof) = n x 20 x 10

        #     clouds = clouds.transpose([0,2,1])

        #     clouds_global = diameters[:,None]
        #     if 'e' in args['global_feat']:
        #         clouds_global = np.hstack((clouds_global, elevation[:,None]))
        #     if 'rgb' in args['global_feat']:
        #         clouds_global = np.hstack((clouds_global, rgb[selected_ver,]))
        #     if 'XY' in args['global_feat']:
        #         clouds_global = np.hstack((clouds_global, xyn))
        #     if 'xy' in args['global_feat']:
        #         clouds_global = np.hstack((clouds_global, xyz[selected_ver,:2]))
        #     if 'o' in args['global_feat']:
        #         clouds_global = np.hstack((clouds_global, geof[selected_ver,]))

        xyz = xyz[selected_ver,]
        rgb = rgb[selected_ver,]

        is_transition = torch.from_numpy(is_transition)
        labels = torch.from_numpy(labels)
        objects = torch.from_numpy(objects.astype('int64'))

        # clouds = torch.from_numpy(clouds)
        # clouds_global = torch.from_numpy(clouds_global)
        # if clouds_global.shape[0] != 15000:
        #     print(short_name)

        xyz = torch.from_numpy(xyz)
        rgb = torch.from_numpy(rgb)

        del raw_labels
        del raw_rgb
        del raw_xyz
        # del nei

        return short_name, edg_source, edg_target, is_transition, labels, objects, xyz, rgb

    def __len__(self):
        return round(len(self.data_idx))


# 提升每一个epoch第一个iteration的速度
class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)
