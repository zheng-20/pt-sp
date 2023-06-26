import torch
import numpy as np
from util.primitive_dis import ComputePrimitiveDistance 
from util.spec_utils import *
from fitting_func import *

def map_type_gt(T_gt):
    T_gt[T_gt == 0] = 9
    T_gt[T_gt == 6] = 9
    T_gt[T_gt == 7] = 9
    T_gt[T_gt == 8] = 2
    return T_gt

def initialize_open_spline_model(mode=0):
    from model.splinenet.splinenet import DGCNNControlPoints
    modelname = 'model/splinenet/pretrained_models/open_spline.pth'

    control_decoder_ = DGCNNControlPoints(20, num_points=10, mode=mode)
    control_decoder = torch.nn.DataParallel(control_decoder_)
    control_decoder.load_state_dict(
        torch.load(modelname)
    )

    control_decoder_.cuda()
    control_decoder_.eval()
    return control_decoder_

def initialize_closed_spline_model(mode=1):
    from model.splinenet.splinenet import DGCNNControlPoints
    modelname = 'model/splinenet/pretrained_models/closed_spline.pth'

    control_decoder_ = DGCNNControlPoints(20, num_points=10, mode=mode)
    control_decoder = torch.nn.DataParallel(control_decoder_)
    control_decoder.load_state_dict(
        torch.load(modelname)
    )

    control_decoder_.cuda()

    control_decoder_.eval()
    return control_decoder_

def forward_pass_open_spline(
        input_points_, control_decoder, nu, nv, viz=False, weights=None, if_optimize=True
):
    nu = nu.cuda(input_points_.get_device())
    nv = nv.cuda(input_points_.get_device())
    with torch.no_grad():
        points_, scales, means, RS = standardize_points_torch(input_points_, weights)

    batch_size = points_.shape[0]
    if viz:
        reg_points = np.copy(points_[:, 0:400])

    # points = Variable(torch.from_numpy(points_.astype(np.float32))).cuda()
    points = points_.permute(0, 2, 1)
    output = control_decoder(points, weights.T)

    # Chamfer Distance loss, between predicted and GT surfaces
    reconstructed_points = sample_points_from_control_points_(
        nu, nv, output, batch_size
    )
    output = output.view(1, 400, 3)

    out_recon_points = []
    new_outputs = []
    for b in range(batch_size):
        # re-alinging back to original orientation for better comparison
        s = scales[b]

        temp = reconstructed_points[b].clone() * s.reshape((1, 3))
        new_points = torch.inverse(RS[b]) @ torch.transpose(temp, 1, 0)
        temp = torch.transpose(new_points, 1, 0)
        temp = temp + means[b]

        out_recon_points.append(temp)

        temp = output[b] * s.reshape((1, 3))
        temp = torch.inverse(RS[b]) @ torch.transpose(temp, 1, 0)
        temp = torch.transpose(temp, 1, 0)
        temp = temp + means[b]
        new_outputs.append(temp)
        if viz:
            new_points = np.linalg.inv(RS[b]) @ reg_points[b].T
            reg_points[b] = new_points.T
            pred_mesh = tessalate_points(reconstructed_points[b], 30, 30)
            gt_mesh = tessalate_points(reg_points[b], 20, 20)
            draw_geometries([pred_mesh, gt_mesh])

    output = torch.stack(new_outputs, 0)
    reconstructed_points = torch.stack(out_recon_points, 0)
    if if_optimize:
        reconstructed_points = optimize_open_spline_kronecker(reconstructed_points, input_points_, output, deform=True)
    return reconstructed_points, reconstructed_points

def forward_closed_splines(input_points_, control_decoder, nu, nv, viz=False, weights=None, if_optimize=True):
    batch_size = input_points_.shape[0]
    nu = nu.cuda(input_points_.get_device())
    nv = nv.cuda(input_points_.get_device())

    with torch.no_grad():
        points_, scales, means, RS = standardize_points_torch(input_points_, weights)

    if viz:
        reg_points = points_[:, 0:400]

    # points = Variable(torch.from_numpy(points_.astype(np.float32))).cuda()
    points = points_.permute(0, 2, 1)
    output = control_decoder(points, weights.T)

    # Chamfer Distance loss, between predicted and GT surfaces
    reconstructed_points = sample_points_from_control_points_(
        nu, nv, output, batch_size
    )

    closed_reconst = []
    closed_control_points = []

    for b in range(batch_size):
        s = scales[b]
        temp = output[b] * s.reshape((1, 3))
        temp = torch.inverse(RS[b]) @ torch.transpose(temp, 1, 0)
        temp = torch.transpose(temp, 1, 0)
        temp = temp + means[b]

        temp = temp.reshape((20, 20, 3))
        temp = torch.cat([temp, temp[0:1]], 0)
        closed_control_points.append(temp)

        temp = (
                reconstructed_points[b].clone() * scales[b].reshape(1, 3)
        )
        temp = torch.inverse(RS[b]) @ temp.T
        temp = torch.transpose(temp, 1, 0) + means[b]
        temp = temp.reshape((30, 30, 3))
        temp = torch.cat([temp, temp[0:1]], 0)
        closed_reconst.append(temp)

    output = torch.stack(closed_control_points, 0)
    reconstructed_points = torch.stack(closed_reconst, 0)
    reconstructed_points = reconstructed_points.reshape((1, 930, 3))

    if if_optimize and (input_points_.shape[1] > 200):
        reconstructed_points = optimize_close_spline_kronecker(reconstructed_points, input_points_, output)
        reconstructed_points = reconstructed_points.reshape((1, 930, 3))
    
    return reconstructed_points, None, reconstructed_points

class FittingModule:
    def __init__(self):
        # get routine for the spline prediction
        nu, nv = uniform_knot_bspline(20, 20, 3, 3, 30)
        self.nu = torch.from_numpy(nu.astype(np.float32))
        self.nv = torch.from_numpy(nv.astype(np.float32))
        self.open_control_decoder = initialize_open_spline_model()
        self.closed_control_decoder = initialize_closed_spline_model()

    def forward_pass_open_spline(self, points, weights=None, if_optimize=False):
        points = torch.unsqueeze(points, 0)

        # NOTE: this will avoid back ward pass through the encoder of SplineNet.
        points.requires_grad = False
        reconst_points = forward_pass_open_spline(
            input_points_=points, control_decoder=self.open_control_decoder, nu=self.nu, nv=self.nv,
            if_optimize=if_optimize, weights=weights)[1]
        # reconst_points = np.array(reconst_points).astype(np.float32)
        torch.cuda.empty_cache()
        return reconst_points

    def forward_pass_closed_spline(self, points, weights=None, if_optimize=False):
        points = torch.unsqueeze(points, 0)
        points.requires_grad = False
        reconst_points = forward_closed_splines(
            points, self.closed_control_decoder, self.nu, self.nv, if_optimize=if_optimize, weights=weights)[2]
        torch.cuda.empty_cache()
        return reconst_points

def construction_affinity_matrix_type(inputs_xyz, type_per_point, T_param_pred, sigma=1.0):

    '''
    inputs_xyz: (B, N, 3)
    embedding: (B, N, 256)
    param_per_point: (B, N, 22)
    type_per_point: (B, N, 10)

    check distance function

    '''

    cp_distance = ComputePrimitiveDistance(reduce=False, one_side=True)
    routines = {5:cp_distance.distance_from_sphere,
                1:cp_distance.distance_from_plane,
                4:cp_distance.distance_from_cylinder,
                3:cp_distance.distance_from_cone,
                2:cp_distance.distance_from_bspline,
                9:cp_distance.distance_from_bspline}
    
    fitter = FittingModule()

    param_list = {5:[0,4], 1:[4,8], 4:[8,15], 3:[15,22]}
    
    if len(type_per_point.shape) == 3:
        T_pred = torch.argmax(type_per_point, dim=-1) # (B, N)
        T_pred = map_type_gt(T_pred)
    else:
        T_pred = map_type_gt(type_per_point)
    B, N = T_pred.shape

    distance_matrix = -torch.ones(B, N, N).float().to(T_pred.device)
    
    for b in range(T_pred.shape[0]):
        type_set = T_pred[b].unique()
        for i in type_set:
            index = torch.where(T_pred[b] == i)[0]
            primitive_type = int(i)
            inputs_xyz_sub = (inputs_xyz[b].T)[index]
            inputs_xyz_ = inputs_xyz[b].T
            
            if primitive_type == -1 or index.shape[0] < 30:
                continue
            
            if primitive_type == 2: # process open spline
                weights = torch.ones([inputs_xyz_sub.shape[0], 1]).to(inputs_xyz_sub.device)
                reconst_points = fitter.forward_pass_open_spline(inputs_xyz_sub, weights=weights)
                
                param_pred = reconst_points
                distance = routines[primitive_type](points=inputs_xyz_, params=param_pred)
                distance = distance[:, None].repeat(1, len(index))
            elif primitive_type == 9: # process closed spline
                weights = torch.ones([inputs_xyz_sub.shape[0], 1]).to(inputs_xyz_sub.device)
                try:
                    reconst_points = fitter.forward_pass_closed_spline(inputs_xyz_sub, weights=weights, if_optimize=True)
                except:
                    reconst_points = fitter.forward_pass_closed_spline(inputs_xyz_sub, weights=weights)
                param_pred = reconst_points
                distance = routines[primitive_type](points=inputs_xyz_, params=param_pred)
                distance = distance[:, None].repeat(1, len(index))
            else:
                param_pred = T_param_pred[b][index] # (M, 22)
                param_pred = param_pred[:, param_list[primitive_type][0]:param_list[primitive_type][1]]
                distance = routines[primitive_type](points=inputs_xyz_, params=param_pred)
            
            distance_matrix[b, :, index] = distance
 
    background_mask = distance_matrix == -1

    affinity_matrix = torch.exp(-distance_matrix**2/ (2 * sigma * sigma))
    affinity_matrix = affinity_matrix + 0
    affinity_matrix[background_mask] = 1e-12
    
    D = affinity_matrix.sum(-1)
    D = torch.diag_embed(1.0 / D.sqrt())
    affinity_matrix = torch.matmul(torch.matmul(D, affinity_matrix), D)
    
    mask = (affinity_matrix > 0).float()
    affinity_matrix = (affinity_matrix + affinity_matrix.permute(
        0, 2, 1)) / (mask + mask.permute(0, 2, 1)).clamp(1, 2)
 
    return affinity_matrix

def construction_affinity_matrix_type_sp(points, normals, type_per_point, pred_sp, t_sp_xyz, sigma=1.0):

    cp_distance = ComputePrimitiveDistance(reduce=False, one_side=True)
    routines = {5:cp_distance.distance_from_sphere,
                1:cp_distance.distance_from_plane,
                4:cp_distance.distance_from_cylinder,
                3:cp_distance.distance_from_cone,
                2:cp_distance.distance_from_bspline,
                9:cp_distance.distance_from_bspline}
    
    fitter = FittingModule()

    n = len(pred_sp)
    distance_matrix = -torch.ones(n, n).float().to(type_per_point.device)

    for i in range(len(pred_sp)):
        p_idx = pred_sp[i][0]
        p_sp = points[p_idx]
        n_sp = normals[p_idx]
        type_per_point_sp = type_per_point[p_idx]
        type_per_point_sp = torch.softmax(type_per_point_sp, dim=-1)
        type_sp = torch.argmax(type_per_point_sp, dim=-1)
        type_sp = map_type_gt(type_sp)
        mode, count = torch.mode(type_sp)   # mode: the most frequent type in the superpoint
        primitive_type = int(mode)
        # weights = np.ones([p_sp.shape[0], 1], dtype=np.float32)
        weights = torch.ones([p_sp.shape[0], 1]).to(p_sp.device)
        N = p_sp.shape[0]
        if N == 1:
            distance = torch.cdist(p_sp, t_sp_xyz).transpose(0, 1)

        # p_sp = torch.from_numpy(p_sp)
        # n_sp = torch.from_numpy(n_sp)
        # weights = torch.from_numpy(weights)

        elif primitive_type == 5: # fitting sphere
            center, radius = fit_sphere_torch(p_sp, n_sp, weights)
            # if radius > 10:
            #     print(filename + ': sphere radius:' + str(radius.cpu().numpy()) + 'point number:' + str(N))
            #     continue

            # center = np.tile(center.cpu().numpy(), [N, 1])
            # radius = np.tile(radius.cpu().numpy(), N)
            # center = torch.tile(center, [N, 1])
            # radius = torch.tile(radius, [N, 1])

            # primitive_param[indx, :3] = center
            # primitive_param[indx, 3] = radius
            param_pred = torch.cat([center, radius.unsqueeze(0).unsqueeze(0)], dim=-1)
            distance = routines[primitive_type](points=t_sp_xyz, params=param_pred)

        elif primitive_type == 1: # fitting plane
            a, d = fit_plane_torch(p_sp, n_sp, weights)
            
            # a = np.tile(a.cpu().numpy(), [N, 1])
            # d = np.tile(d.cpu().numpy(), N)
            # a = torch.tile(a, [N, 1])
            # d = torch.tile(d, [N, 1])

            # primitive_param[indx, 4:7] = a
            # primitive_param[indx, 7] = d
            param_pred = torch.cat([a, d.unsqueeze(0).unsqueeze(0)], dim=-1)
            distance = routines[primitive_type](points=t_sp_xyz, params=param_pred)


        elif primitive_type == 4: # fitting cylinder
            try:
                a, center, radius = fit_cylinder_torch(p_sp, n_sp, weights)
            except:
                print('raise error', 'point number:' + str(N) + '\n')
                continue

            # if radius > 10 or center[0][0] > 10 or center[0][1] > 10 or center[0][2] > 10:
            #     print(filename + ': cylinder radius:' + str(radius) +'point number:' + str(N))
            #     continue
            
            # a = np.tile(a.T.cpu().numpy(), [N, 1])
            # center = np.tile(center, [N, 1])
            # radius = np.tile(radius, N)
            # a = torch.tile(a.T, [N, 1])
            center = torch.from_numpy(center).to(a.device)
            radius = torch.from_numpy(np.array(radius)).to(a.device)

            # primitive_param[indx, 8:11] = a
            # primitive_param[indx, 11:14] = center
            # primitive_param[indx, 14] = radius
            param_pred = torch.cat([a.T, center, radius.unsqueeze(0).unsqueeze(0)], dim=-1)
            distance = routines[primitive_type](points=t_sp_xyz, params=param_pred)

        elif primitive_type == 3:
            center, a, theta = fit_cone_torch(p_sp, n_sp, weights)
            # if center[0][0] > 10 or center[1][0] > 10 or center[2][0] > 10:
            #     tmp = center.cpu().numpy()
            #     print(filename + ': cone center:' + str(tmp[0][0]) + ' '+ str(tmp[1][0]) + ' ' + str(tmp[2][0]) +'point number:' + str(N))
            #     continue
            
            # a = np.tile(a.cpu().numpy(), [N, 1])
            # center = np.tile(center.T.cpu().numpy(), [N, 1])
            # theta = np.tile(theta.cpu().numpy(), [N, 1])
            # a = torch.tile(a, [N, 1])
            # center = torch.tile(center.T, [N, 1])
            # theta = torch.tile(theta, [N, 1])

            # primitive_param[indx, 15:18] = a
            # primitive_param[indx, 18:21] = center
            # primitive_param[indx, 21] = theta
            if theta.dim() == 1:
                theta = theta.unsqueeze(0)
            else:
                theta = theta.unsqueeze(0).unsqueeze(0)
            param_pred = torch.cat([a, center.T, theta], dim=-1).to(t_sp_xyz.device)
            distance = routines[primitive_type](points=t_sp_xyz, params=param_pred)

        elif primitive_type == 2: # process open spline
            weights = torch.ones([p_sp.shape[0], 1]).to(p_sp.device)
            reconst_points = fitter.forward_pass_open_spline(p_sp, weights=weights)
            
            param_pred = reconst_points
            distance = routines[primitive_type](points=t_sp_xyz, params=param_pred)

        elif primitive_type == 9: # process closed spline
            weights = torch.ones([p_sp.shape[0], 1]).to(p_sp.device)
            try:
                reconst_points = fitter.forward_pass_closed_spline(p_sp, weights=weights, if_optimize=True)
            except:
                reconst_points = fitter.forward_pass_closed_spline(p_sp, weights=weights)
            param_pred = reconst_points
            distance = routines[primitive_type](points=t_sp_xyz, params=param_pred)

        distance_matrix[:, i] = distance.squeeze()

    background_mask = distance_matrix == -1

    affinity_matrix = torch.exp(-distance_matrix**2/ (2 * sigma * sigma))
    affinity_matrix = affinity_matrix + 0
    affinity_matrix[background_mask] = 1e-12
    
    D = affinity_matrix.sum(-1)
    D = torch.diag_embed(1.0 / D.sqrt())
    affinity_matrix = torch.matmul(torch.matmul(D, affinity_matrix), D)
    
    mask = (affinity_matrix > 0).float()
    affinity_matrix = (affinity_matrix + affinity_matrix.permute(
        0, 1)) / (mask + mask.permute(0, 1)).clamp(1, 2)
 
    return affinity_matrix



def find_closest_xyz(sp_xyz, txyz, pred_components):
    '''
    查找每个超点中，距离平均点最近的点
    :param sp_xyz: 超点平均点的xyz
    :param txyz: 模型所有点的xyz
    :param pred_components: 超点的索引
    :return: 超点中距离平均点最近的点的xyz, shape: [sp_num, 3]
    '''
    t_sp_xyz = torch.ones([sp_xyz.shape[0], 3]).to(sp_xyz.device)
    for i in range(len(pred_components)):
        p_xyz = txyz[pred_components[i][0]]
        dist = torch.cdist(sp_xyz[i].unsqueeze(0), p_xyz)
        sp_xyz_ = p_xyz[torch.argmin(dist)]
        t_sp_xyz[i] = sp_xyz_
    
    return t_sp_xyz


def compute_entropy(features):
    '''
    features: (1, N, K) K = dim of feature
    '''
    
    eps = 1e-7
    assert(features.shape[0] == 1)

    
    feat = features[0]

    N, K = feat.shape
    
    average_dst = 0
    
    # calculate interval
    max_list = []
    min_list = []
    
    # save gpu memory
    for i in range(7):
        for j in range(7):
            max_ = torch.max((feat[i*1000:(i+1)*1000, None, :] - feat[None, j*1000:(j+1)*1000, :]).view(-1,K), dim=0)[0][None, :]   # (1, K)
            min_ = torch.min((feat[i*1000:(i+1)*1000, None, :] - feat[None, j*1000:(j+1)*1000, :]).view(-1,K), dim=0)[0][None, :]   # (1, K)
            max_list.append(max_)
            min_list.append(min_)
    
    max_all = torch.max(torch.cat(max_list, dim=0), dim=0)[0]   # (K)
    min_all = torch.min(torch.cat(min_list, dim=0), dim=0)[0]   # (K)
    interval = max_all - min_all

    # calculate average_dst
    for i in range(7):
        for j in range(7):
            dst = torch.norm((feat[i*1000:(i+1)*1000, None, :] - feat[None, j*1000:(j+1)*1000, :]) / interval, dim=2)

            average_dst += torch.sum(dst)
    
    average_dst /= (N*N)
    
    alpha = -np.log(0.5) / average_dst
    
    E = 0

    for i in range(7):
        for j in range(7):
            dst = torch.norm((feat[i*1000:(i+1)*1000, None, :] - feat[None, j*1000:(j+1)*1000, :]) / interval, dim=2)
            s = torch.exp(-alpha * dst)

            entropy = - s * torch.log(s + eps) - (1 - s) * torch.log(1 - s + eps)

            E += torch.sum(entropy)

    E /= (N*N)

    return E


def compute_entropy_sp(features):
    
    eps = 1e-7
    # assert(features.shape[0] == 1)

    feat = features

    N, K = feat.shape

    average_dst = 0

    # calculate interval
    max_list = []
    min_list = []

    bs = 5
    remainder = N % bs
    l = N // bs

    if remainder != 0:
        iter = bs + 1
    else:
        iter = bs

    for i in range(iter):
        for j in range(iter):
            max_ = torch.max((feat[i*l:(i+1)*l, None, :] - feat[None, j*l:(j+1)*l, :]).view(-1,K), dim=0)[0][None, :]
            min_ = torch.min((feat[i*l:(i+1)*l, None, :] - feat[None, j*l:(j+1)*l, :]).view(-1,K), dim=0)[0][None, :]
            max_list.append(max_)
            min_list.append(min_)
    
    max_all = torch.max(torch.cat(max_list, dim=0), dim=0)[0]
    min_all = torch.min(torch.cat(min_list, dim=0), dim=0)[0]
    interval = max_all - min_all

    # calculate average_dst
    for i in range(iter):
        for j in range(iter):
            dst = torch.norm((feat[i*l:(i+1)*l, None, :] - feat[None, j*l:(j+1)*l, :]) / interval, dim=2)

            average_dst += torch.sum(dst)

    average_dst /= (N*N) 

    alpha = -np.log(0.5) / average_dst

    E = 0

    for i in range(iter):
        for j in range(iter):
            dst = torch.norm((feat[i*l:(i+1)*l, None, :] - feat[None, j*l:(j+1)*l, :]) / interval, dim=2)
            s = torch.exp(-alpha * dst)

            entropy = - s * torch.log(s + eps) - (1 - s) * torch.log(1 - s + eps)

            E += torch.sum(entropy)
    
    E /= (N*N)

    return E

    



def npy(var):
    return var.data.cpu().numpy()