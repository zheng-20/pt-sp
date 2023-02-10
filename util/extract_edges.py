from multiprocessing import Pool
import os
import numpy as np
# import trimesh
import time

# def extract(i):
#     data_root = '/home/fz20/dataset/ABC/train_final/'
#     write_path = '/home/fz20/dataset/ABC/train/'
#     start = time.time()
#     data_path = os.path.join(data_root, i)
#     data = np.load(data_path)
#     coord, normals, boundary, label, semantic, param, F = data['V'],data['N'],data['B'],data['L'],data['S'],data['T_param'],data['F']
#     edges = trimesh.geometry.faces_to_edges(F)
#     edges_list = []
#     for j in range(len(coord)):
#         n, _ = np.where(edges == j)
#         nn_idx = np.unique(edges[n][edges[n] != j])
#         edges_list.append(nn_idx)
#     edges_list = np.array(edges_list, dtype=object)
#     np.savez(write_path + i, V=coord, N=normals, B=boundary, L=label, S=semantic, T_param=param, F=F, edges=edges_list)
#     print(i)
#     # print("time for {}: {}s".format(count, time.time()-start))

    
# data_root = '/home/fz20/dataset/ABC/train_final/'
# data_list = sorted(os.listdir(data_root))
# # count = 1
# pool = Pool(processes=32)
# pool.map(extract, data_list)



# data_root = '/home/fz20/dataset/ABC_edges/train_final/'
# write_path = '/data/fz20/dataset/ABC/train/'
# data_list = sorted(os.listdir(data_root))
# mmax = 0
# for i in data_list:
def ff(i):
    data_root = '/home/fz20/dataset/ABC_edges/val_final/'
    write_path = '/data/fz20/dataset/ABC/val/'
    data_path = os.path.join(data_root, i)
    data = np.load(data_path, allow_pickle=True)
    coord, normals, boundary, label, semantic, param, F, edges = data['V'],data['N'],data['B'],data['L'],data['S'],data['T_param'],data['F'],data['edges']
    edges_list = []
    # max = 0
    for j in edges:
        ee = []
        ee.append(len(j))
        for k in j:
            ee.append(k)
        if len(ee) < 47:
            for m in range(47-len(ee)):
                ee.append(-1)
        edges_list.append(ee)
    edges_list = np.array(edges_list, dtype=np.int32)
        # if len(j) > max:
        #     max = len(j)
    # if max > mmax:
    #     mmax = max

    np.savez(write_path + i, V=coord, N=normals, B=boundary, L=label, S=semantic, T_param=param, F=F, edges=edges_list)
    print(i)

data_root = '/home/fz20/dataset/ABC_edges/val_final/'
data_list = sorted(os.listdir(data_root))
# count = 1
pool = Pool(processes=40)
pool.map(ff, data_list)