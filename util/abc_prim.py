import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../'))
import numpy as np
# import SharedArray as SA

import torch
from torch.utils.data import Dataset

from util.voxelize import voxelize
from util.data_util import sa_create, collate_fn, dataAugment, data_prepare_abc, data_prepare_dse_abc
# from util.data_util import data_prepare_v101 as data_prepare
# import open3d as o3d
from lib.boundaryops.functions import boundaryops



class ABC_Dataset(Dataset):
    def __init__(self, split='train', data_root='trainval', voxel_size=0.04, voxel_max=None, shuffle_index=False, loop=1):
        super().__init__()
        # self.split, self.voxel_size, self.transform, self.voxel_max, self.shuffle_index, self.loop = split, voxel_size, transform, voxel_max, shuffle_index, loop
        self.split, self.loop, self.voxel_size = split, loop, voxel_size
        if split == 'train':
            data_root += '/train_final/'
        elif split == 'val':
            data_root += '/val_final/'
        data_list = sorted(os.listdir(data_root))
        self.data_list = [item[:-4] for item in data_list]
        # if split == 'train':
        #     self.data_list = [item for item in data_list if not 'Area_{}'.format(test_area) in item]
        # else:
        #     self.data_list = [item for item in data_list if 'Area_{}'.format(test_area) in item]
        self.data_root = data_root
        # for item in self.data_list:
        #     if not os.path.exists("/dev/shm/{}".format(item)):
        #         data_path = os.path.join(data_root, item + '.npy')
        #         data = np.load(data_path)  # xyzrgbl, N*7
        #         sa_create("shm://{}".format(item), data)
        self.data_idx = np.arange(len(self.data_list))
        print("Totally {} samples in {} set.".format(len(self.data_idx), split))

    def __getitem__(self, idx):
        data_idx = self.data_idx[idx % len(self.data_idx)]

        # data = SA.attach("shm://{}".format(self.data_list[data_idx])).copy()
        item = self.data_list[data_idx]
        data_path = os.path.join(self.data_root, item + '.npz')
        data = np.load(data_path)

        coord, normals, boundary, label, semantic, param, F, edges = data['V'],data['N'],data['B'],data['L'],data['S'],data['T_param'],data['F'],data['edges']
        coord, normals, boundary, label, semantic, param, F, edges = data_prepare_abc(coord, normals, boundary, label, semantic, param, F, edges, voxel_size=self.voxel_size)
        # coord, normals = dataAugment(coord, normals, False, True, True)
        # coord, feat, label = data[:, 0:3], data[:, 3:6], data[:, 6]
        # coord, feat, label = data_prepare(coord, feat, label, self.split, self.voxel_size, self.voxel_max, self.transform, self.shuffle_index)

        return coord, normals, boundary, label, semantic, param, F, edges

    def __len__(self):
        return round(len(self.data_idx) * self.loop)
    

class ABC_dse_Dataset(Dataset):
    def __init__(self, split='train', data_root='trainval', voxel_size=0.04, voxel_max=None, shuffle_index=False, loop=1):
        super().__init__()
        # self.split, self.voxel_size, self.transform, self.voxel_max, self.shuffle_index, self.loop = split, voxel_size, transform, voxel_max, shuffle_index, loop
        self.split, self.loop, self.voxel_size = split, loop, voxel_size
        if split == 'train':
            data_root += '/train_final/'
        elif split == 'val':
            data_root += '/val_final/'
        data_list = sorted(os.listdir(data_root))
        self.data_list = [item[:-4] for item in data_list]
        # if split == 'train':
        #     self.data_list = [item for item in data_list if not 'Area_{}'.format(test_area) in item]
        # else:
        #     self.data_list = [item for item in data_list if 'Area_{}'.format(test_area) in item]
        self.data_root = data_root
        # for item in self.data_list:
        #     if not os.path.exists("/dev/shm/{}".format(item)):
        #         data_path = os.path.join(data_root, item + '.npy')
        #         data = np.load(data_path)  # xyzrgbl, N*7
        #         sa_create("shm://{}".format(item), data)
        self.data_idx = np.arange(len(self.data_list))
        print("Totally {} samples in {} set.".format(len(self.data_idx), split))

    def __getitem__(self, idx):
        data_idx = self.data_idx[idx % len(self.data_idx)]

        # data = SA.attach("shm://{}".format(self.data_list[data_idx])).copy()
        item = self.data_list[data_idx]
        data_path = os.path.join(self.data_root, item + '.npz')
        data = np.load(data_path)

        coord, normals, boundary, label, semantic, param, F, edges, dse_edges = data['V'],data['N'],data['B'],data['L'],data['S'],data['T_param'],data['F'],data['edges'],data['dse_edges']
        coord, normals, boundary, label, semantic, param, F, edges, dse_edges = data_prepare_dse_abc(coord, normals, boundary, label, semantic, param, F, edges, dse_edges, voxel_size=self.voxel_size)
        # coord, normals = dataAugment(coord, normals, False, True, True)
        # coord, feat, label = data[:, 0:3], data[:, 3:6], data[:, 6]
        # coord, feat, label = data_prepare(coord, feat, label, self.split, self.voxel_size, self.voxel_max, self.transform, self.shuffle_index)

        return coord, normals, boundary, label, semantic, param, F, edges, dse_edges

    def __len__(self):
        return round(len(self.data_idx) * self.loop)


if __name__ == '__main__':
    data_root = '/data/fz20/dataset/ABC'
    test_area, voxel_size, voxel_max = 5, 0.005, 80000

    point_data = ABC_Dataset(split='train', data_root=data_root, voxel_size=voxel_size, voxel_max=voxel_max)
    print('point data size:', point_data.__len__())
    import torch, time, random
    manual_seed = 123
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    def worker_init_fn(worker_id):
        random.seed(manual_seed + worker_id)
    train_loader = torch.utils.data.DataLoader(point_data, batch_size=3, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate_fn)
    for idx in range(1):
        end = time.time()
        voxel_num = []
        for i, (coord, normals, boundary, label, semantic, param, offset, edges) in enumerate(train_loader):
            coord, normals, boundary, label, semantic, param, offset, edges = coord.cuda(non_blocking=True), normals.cuda(non_blocking=True), boundary.cuda(non_blocking=True), \
                                label.cuda(non_blocking=True), semantic.cuda(non_blocking=True), param.cuda(non_blocking=True), offset.cuda(non_blocking=True), edges.cuda(non_blocking=True)
            start = time.time()
            boundaryops.boundaryquery(8, coord, coord, offset, offset, edges, boundary)
            print(time.time()-start)

            if boundary.max() > 1 or boundary.min() < 0:
                import ipdb
                ipdb.set_trace()
            if semantic.max() > 9 or semantic.min() < 0:
                import ipdb
                ipdb.set_trace()
            print('time: {}/{}--{}'.format(i+1, len(train_loader), time.time() - end))
            # print('tag', coord.shape, normals.shape, label.shape, offset.shape, torch.unique(label))
            voxel_num.append(label.shape[0])
            end = time.time()
    print(np.sort(np.array(voxel_num)))
