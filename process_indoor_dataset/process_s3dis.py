import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../'))
sys.path.append(os.path.join(BASE_DIR, '../util'))
import numpy as np
import h5py
import open3d as o3d
import torch
from util.fitting_func import *
from util.primitive_dis import ComputePrimitiveDistance
import time
from tqdm import *
from multiprocessing import Pool

def write_structure(file_name, xyz, rgb, source, target,  target_local_geometry, is_transition, labels, objects, geof, elevation, xyn, normals, primitive_param):
    """
    save the input point cloud in a format ready for embedding    
    """
    #store transition and non-transition edges in two different contiguous memory blocks
    #n_transition = np.count_nonzero(is_transition)
    #blocks = np.hstack((np.where(is_transition),np.where(is_transition==False)))
    
    data_file = h5py.File(file_name, 'w')
    data_file.create_dataset('xyz', data=xyz, dtype='float32')
    data_file.create_dataset('rgb', data=rgb, dtype='float32')
    data_file.create_dataset('elevation', data=elevation, dtype='float32')
    data_file.create_dataset('xyn', data=xyn, dtype='float32')
    data_file.create_dataset('source', data=source, dtype='int')
    data_file.create_dataset('target', data=target, dtype='int')
    data_file.create_dataset('is_transition', data=is_transition, dtype='uint8')
    data_file.create_dataset('target_local_geometry', data=target_local_geometry, dtype='uint32')
    data_file.create_dataset('objects', data=objects, dtype='uint32')
    data_file.create_dataset('T_param', data=primitive_param, dtype='float32')
    data_file.create_dataset('normals', data=normals, dtype='float32')
    if (len(geof)>0):        
        data_file.create_dataset('geof', data=geof, dtype='float32')
    if len(labels) > 0 and len(labels.shape)>1 and labels.shape[1]>1:
        data_file.create_dataset('labels', data=labels, dtype='int32')
    else:
        data_file.create_dataset('labels', data=labels, dtype='uint8')

def fit_scene(data_dir, write_dir):
    start = time.time()
    # data_dir = '/data/fz20/dataset/S3DIS/Stanford3dDataset_v1.2_Aligned_Version/features_supervision/Area_1/conferenceRoom_1.h5'
    # write_dir = '/data1/fz20/dataset/sp_s3dis_dataset/Area_1'
    str_file = os.path.join(write_dir, os.path.basename(data_dir))
    if os.path.exists(str_file):
        return 0

    data_file = h5py.File(data_dir, 'r')
    xyz = np.array(data_file['xyz'], dtype='float32')
    rgb = np.array(data_file['rgb'], dtype='float32')
    elevation = np.array(data_file['elevation'], dtype='float32')
    xyn = np.array(data_file['xyn'], dtype='float32')
    edg_source = np.array(data_file['source'], dtype='int').squeeze()
    edg_target = np.array(data_file['target'], dtype='int').squeeze()
    is_transition = np.array(data_file['is_transition'])
    objects = np.array(data_file['objects'][()])
    labels = np.array(data_file['labels']).squeeze()
    l = np.argmax(labels, axis=1)
    if len(labels.shape) == 0:
        labels = np.array([0])
    if len(is_transition.shape) == 0:
        is_transition = np.array([0])
    # if read_geof:
    #     local_geometry = np.array(data_file['geof'], dtype='float32')
    # else:
    local_geometry = np.array(data_file['target_local_geometry'], dtype='uint32')
    geof = np.array(data_file['geof'], dtype='float32')
    data_file.close()

    pcd = o3d.geometry.PointCloud() # 定义点云
    pcd.points = o3d.utility.Vector3dVector(xyz)
    # pcd.colors = o3d.utility.Vector3dVector(rgb)
    pcd.estimate_normals()  # 估计法向量
    pcd.orient_normals_consistent_tangent_plane(30) # 使法向量朝向一致

    normal = np.asarray(pcd.normals, dtype='float32')

    cp_distance = ComputePrimitiveDistance(reduce=False, one_side=True) # 定义距离计算函数

    primitive_param = torch.zeros((xyz.shape[0], 15), dtype=torch.float32)
    # render shape parameters
    # 0:sphere 4dim, 1:plane 4dim, 2:cylinder 7dim, 3:cone 7dim
    # 4 + 4 + 7 + 7 = 22
    # 定义ground truth参数，不包含圆锥（因为室内场景一般拟合不出圆锥的参数）

    # for i in range(max(objects)+1): # 输出每个物体的点数
    #     obj_xyz = xyz[objects == p_idx]
    #     print(obj_xyz.shape[0])

    for i in range(max(objects)): # 遍历每个物体
        p_idx = i + 1
        obj_xyz = xyz[objects == p_idx]   # 选出该物体的点云
        if obj_xyz.shape[0] == 0:   # 如果该物体点数为0，跳过
            continue

        obj_normal = normal[objects == p_idx] # 选出该物体的法向量
        obj_pcd = o3d.geometry.PointCloud()
        obj_pcd.points = o3d.utility.Vector3dVector(obj_xyz)
        obj_pcd.normals = o3d.utility.Vector3dVector(obj_normal)
        # o3d.visualization.draw_geometries([obj_pcd], point_show_normal=True)

        # 为了加快速度，对于平面类别的物体，直接拟合平面参数
        obj_label = np.argmax(np.bincount(l[objects == p_idx]))
        plane_label = [1,2,3,6,8,12]    # 平面类别(天花板、地板、墙、窗户、桌子、黑板)
        if obj_label in plane_label:
            fit_xyz = torch.from_numpy(obj_xyz)
            fit_normal = torch.from_numpy(obj_normal)
            fit_weights = torch.ones((fit_xyz.shape[0], 1), dtype=torch.float32)
            a, d = fit_plane_torch(fit_xyz, fit_normal, fit_weights)
            param_pred_plane = torch.cat([a, d.unsqueeze(0).unsqueeze(0)], dim=-1)
            primitive_param[objects == p_idx, 4:8] = param_pred_plane
        elif obj_xyz.shape[0] < 100:   # 对于点数小于100的物体，直接使用全部点拟合
            fit_xyz = torch.from_numpy(obj_xyz)
            fit_normal = torch.from_numpy(obj_normal)
            fit_weights = torch.ones((fit_xyz.shape[0], 1), dtype=torch.float32)
            
            # 拟合平面
            a, d = fit_plane_torch(fit_xyz, fit_normal, fit_weights)
            param_pred_plane = torch.cat([a, d.unsqueeze(0).unsqueeze(0)], dim=-1)
            distance_plane = cp_distance.distance_from_plane(points=fit_xyz, params=param_pred_plane).mean().unsqueeze(0)

            # 拟合球体
            center, radius = fit_sphere_torch(fit_xyz, fit_normal, fit_weights)
            param_pred_sphere = torch.cat([center, radius.unsqueeze(0).unsqueeze(0)], dim=-1)
            distance_sphere = cp_distance.distance_from_sphere(points=fit_xyz, params=param_pred_sphere).mean().unsqueeze(0)

            # 拟合圆柱体
            try:
                a, center, radius = fit_cylinder_torch(fit_xyz, fit_normal, fit_weights)
            except:
                print('raise error', 'point number:' + str(fit_xyz.shape[0]) + '\n')
                continue
            param_pred_cylinder = torch.cat([a.T, torch.from_numpy(center), torch.from_numpy(np.array(radius)).unsqueeze(0).unsqueeze(0)], dim=-1)
            if param_pred_cylinder.dtype == torch.float64:
                param_pred_cylinder = param_pred_cylinder.float()
            distance_cylinder = cp_distance.distance_from_cylinder(points=fit_xyz, params=param_pred_cylinder).mean().unsqueeze(0)

            primitive_type_idx = torch.argmin(torch.cat([distance_plane, distance_sphere, distance_cylinder]))   # 选出距离最小的拟合结果
            if primitive_type_idx == 0:
                primitive_type = 'plane'
                primitive_param[objects == p_idx, 4:8] = param_pred_plane
            elif primitive_type_idx == 1:
                primitive_type = 'sphere'
                primitive_param[objects == p_idx, :4] = param_pred_sphere
            elif primitive_type_idx == 2:
                primitive_type = 'cylinder'
                primitive_param[objects == p_idx, 8:15] = param_pred_cylinder

        else:

            # 对于非平面类别的物体，使用knn的点拟合四种不同的几何参数，取最优的一种
            # for j in range(obj_xyz.shape[0]):   # 遍历该物体的每个点
            obj_pcd_tree = o3d.geometry.KDTreeFlann(obj_pcd)    # 构建该物体的KDTree
            pcd_tree = o3d.geometry.KDTreeFlann(pcd)    # 构建当前室内场景点云的KDTree
            with trange(obj_xyz.shape[0]) as t:
                for j in t:
                    t.set_description('Processing %s' % str(p_idx))
                    [obj_k, obj_idx, obj_dis] = obj_pcd_tree.search_knn_vector_3d(obj_pcd.points[j], int(obj_xyz.shape[0] / 40))    # 搜索该物体中距离该点最近的k个点
                    [k, idx, dis] = pcd_tree.search_knn_vector_3d(obj_pcd.points[j], 1)   # 搜索该点在当前室内场景中点云的索引

                    fit_xyz = torch.from_numpy(obj_xyz[obj_idx, :]) # 选出该物体中距离该点最近的k个点
                    fit_normal = torch.from_numpy(obj_normal[obj_idx, :])   # 选出该物体中距离该点最近的k个点的法向量
                    fit_weights = torch.ones([fit_xyz.shape[0], 1], dtype=torch.float32)    # 定义该物体中距离该点最近的k个点的权重

                    # 拟合平面
                    a, d = fit_plane_torch(fit_xyz, fit_normal, fit_weights)
                    param_pred_plane = torch.cat([a, d.unsqueeze(0).unsqueeze(0)], dim=-1)
                    distance_plane = cp_distance.distance_from_plane(points=fit_xyz, params=param_pred_plane).mean().unsqueeze(0)

                    # 拟合球体
                    center, radius = fit_sphere_torch(fit_xyz, fit_normal, fit_weights)
                    param_pred_sphere = torch.cat([center, radius.unsqueeze(0).unsqueeze(0)], dim=-1)
                    distance_sphere = cp_distance.distance_from_sphere(points=fit_xyz, params=param_pred_sphere).mean().unsqueeze(0)

                    # 拟合圆柱体
                    try:
                        a, center, radius = fit_cylinder_torch(fit_xyz, fit_normal, fit_weights)
                    except:
                        print('raise error', 'point number:' + str(fit_xyz.shape[0]) + '\n')
                        continue
                    param_pred_cylinder = torch.cat([a.T, torch.from_numpy(center), torch.from_numpy(np.array(radius)).unsqueeze(0).unsqueeze(0)], dim=-1)
                    distance_cylinder = cp_distance.distance_from_cylinder(points=fit_xyz, params=param_pred_cylinder).mean().unsqueeze(0)

                    # # 拟合圆锥体
                    # center, a, theta = fit_cone_torch(fit_xyz, fit_normal, fit_weights)
                    # param_pred_cone = torch.cat([a, center.T, theta.unsqueeze(0).unsqueeze(0)], dim=-1)
                    # distance_cone = cp_distance.distance_from_cone(points=fit_xyz, params=param_pred_cone).mean().unsqueeze(0)

                    primitive_type_idx = torch.argmin(torch.cat([distance_plane, distance_sphere, distance_cylinder]))   # 选出距离最小的拟合结果
                    if primitive_type_idx == 0:
                        primitive_type = 'plane'
                        primitive_param[idx[0], 4:8] = param_pred_plane
                    elif primitive_type_idx == 1:
                        primitive_type = 'sphere'
                        primitive_param[idx[0], :4] = param_pred_sphere
                    elif primitive_type_idx == 2:
                        primitive_type = 'cylinder'
                        primitive_param[idx[0], 8:15] = param_pred_cylinder
                    # elif primitive_type_idx == 3:
                    #     primitive_type = 'cone'
                    #     primitive_param[idx[0], 15:] = param_pred_cone
                    # print('第{}个物体的第{}个点拟合类型为：{}'.format(p_idx, j+1, primitive_type))
                    # t.set_postfix(primitive_type=primitive_type)

    # str_file = '/data1/fz20/dataset/sp_s3dis_dataset/Area_1/conferenceRoom_1.h5'
    write_structure(str_file, xyz, rgb, edg_source, edg_target, local_geometry, is_transition, labels, objects, geof, elevation, xyn, normal, primitive_param)

    # 输出用时
    end = time.time()
    print('{} time cost {}: '.format(os.path.basename(data_dir), end - start))


if __name__ == '__main__':

    raw_data_dir = '/data/fz20/dataset/S3DIS/Stanford3dDataset_v1.2_Aligned_Version/features_supervision/Area_3'
    new_data_dir = '/data1/fz20/dataset/sp_s3dis_dataset/Area_3'

    raw_file_list = [os.path.join(raw_data_dir, file) for file in sorted(os.listdir(raw_data_dir)) if file.endswith('.h5')]
    # new_file_list = [os.path.join(new_data_dir, file) for file in sorted(os.listdir(raw_data_dir)) if file.endswith('.h5')]

    # pool = Pool(processes=2)
    # pool.map(fit_scene, raw_file_list)
    # pool.close()
    # pool.join()

    for scene in raw_file_list:
        fit_scene(scene, new_data_dir)
    

