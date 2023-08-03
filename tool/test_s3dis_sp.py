"""
NOTE: 
"""

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../'))
import time
import random
import numpy as np
import logging
import argparse
import importlib

import torch
# import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.optim.lr_scheduler as lr_scheduler

# from util import dataset, transform, config
import yaml
from util.common_util import AverageMeter, intersectionAndUnionGPU

import torchnet as tnt
import util.metrics as metrics
from util import config
from util.metrics import *
from util.sp_util import get_components, perfect_prediction, relax_edge_binary, partition2ply
# from util.util import perfect_prediction_base0
# from util.util import prediction2ply_seg

# from util.util import write_spg
# from util.graphs import compute_sp_graph_yj

# from util.provider import *

# from util.S3DIS_dataset import create_s3dis_datasets, my_collate
# from util.VKITTI_dataset import create_vkitti_datasets, my_collate_vkitti
# from util.SCANNET_dataset import create_scannet_datasets, my_collate_scannet
# from util.SCANNET_dataset_v1 import create_scannet_datasets_v1, my_collate_scannet_v1
from util.sp_S3DIS_dataset import create_s3dis_datasets, collate_s3dis, s3dis_Dataset, MultiEpochsDataLoader
from util.graphs import compute_sp_graph
from util.provider import write_spg

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Point Cloud Superpoint Generation')
    parser.add_argument('--config', type=str, default=None, required=True, help='config file')
    parser.add_argument('--model_path', type=str, default=None, required=True, help='save path')
    parser.add_argument('--save_folder', type=str, default=None, required=True, help='save_folder path')
    # parser.add_argument('--epoch', type=int, default=None, required=True, help='corresponding to the train_epoch_xx.pth')

    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    cfg["model_path"] = args.model_path
    cfg["save_folder"] = args.save_folder
    # cfg["epoch"] = args.epoch
    
    print("#"*20)
    print("Parameters:")
    for ky in cfg.keys():
        print('key: {} -> {}'.format(ky, cfg[ky]))
    print("#"*20)
    return cfg

def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger

def worker_init_fn(worker_id):
    random.seed(args["manual_seed"] + worker_id)

def init():
    global args, logger
    args = get_parser()

    logger = get_logger()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args["test_gpu"])
    logger.info(args)


def main():
    init()
    
    # MODEL = importlib.import_module(args["arch"])  # import network module
    if args.arch == 'superpoint_net':
        from model.superpoint.superpoint_net import superpoint_seg_repro as Model
    elif args.arch == 'superpoint_fcn_net':
        from model.superpoint.superpoint_net import superpoint_fcn_seg_repro as Model
    elif args.arch == 'PSPT':
        from model.superpoint.superpoint_net import SuperpointNetwork as Model
    else:
        raise Exception('architecture {} not supported yet'.format(args.arch))
    logger.info("load {}.py success!".format(args["arch"]))

    model = Model(c=args.fea_dim, k=args.classes, args=args)
    
    total = sum([param.nelement() for param in model.parameters()])
    logger.info("Number of params: %.2fM" % (total/1e6))

    
    if args["sync_bn"]:
        # from util.util import convert_to_syncbn
        # convert_to_syncbn(model) #, patch_replication_callback(model)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()

    criterion = nn.CrossEntropyLoss(ignore_index=args["ignore_label"]).cuda()
    if args['re_xyz_loss'] == 'mse':
        criterion_re_xyz = nn.MSELoss().cuda()
    else:
        print('re_xyz_loss type error')
        exit()
    
    if args['re_label_loss'] == 'cel':
        criterion_re_label = nn.CrossEntropyLoss(ignore_index=args["ignore_label"]).cuda()
    elif args['re_label_loss'] == 'mse':
        criterion_re_label = nn.MSELoss().cuda()
    else:
        print('re_label_loss type error')
        exit()

    if args['re_sp_loss'] == 'cel':
        criterion_re_sp = nn.CrossEntropyLoss(ignore_index=args["ignore_label"]).cuda()
    elif args['re_sp_loss'] == 'mse':
        criterion_re_sp = nn.MSELoss().cuda()
    else:
        print('re_label_loss type error')
        exit()

    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args["classes"]))
    logger.info(model)
    model = torch.nn.DataParallel(model.cuda())
    
    model_path = os.path.join(args["model_path"], "model", "model_best.pth")
    # model_path = os.path.join(args["model_path"], "model", "model_best_asa.pth")
    if os.path.isfile(model_path):
        logger.info("=> loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path)
        epoch = checkpoint['epoch']
        # model.load_state_dict(checkpoint['state_dict'], strict=False)
        model.load_state_dict(checkpoint['state_dict'])
        logger.info("=> loaded checkpoint '{}'".format(model_path))
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(model_path))
    

    if args["data_name"] == 's3dis':
        train_data = s3dis_Dataset(args, split='train')
        test_data = s3dis_Dataset(args, split='test')
        collate_fn = collate_s3dis
    # elif args['data_name'] == 'vkitti':
    #     train_data, test_data = create_vkitti_datasets(args, logger)
    #     my_collate_s = my_collate_vkitti
    # elif args['data_name'] == 'scannet':
    #     train_data, test_data = create_scannet_datasets(args, logger)
    #     my_collate_s = my_collate_scannet
    # elif args['data_name'] == 'scannet_v1':
    #     train_data, test_data = create_scannet_datasets_v1(args, logger)
    #     my_collate_s = my_collate_scannet_v1
    else:
        print("data_name error")
        exit()
   
    # test_loader = torch.utils.data.DataLoader(test_data,
    #                                            batch_size=args["batch_size_val"],
    #                                            shuffle=False,
    #                                            num_workers=args["workers"],
    #                                            collate_fn=collate_fn,
    #                                            pin_memory=True,
    #                                            drop_last=False)
    test_loader = MultiEpochsDataLoader(test_data, batch_size=args.batch_size_test, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=False, collate_fn=collate_fn)

    test(test_loader, model, criterion, criterion_re_xyz, criterion_re_label, criterion_re_sp, args.epochs)

def label2one_hot(labels, C=10):
    n = labels.size(0)
    labels = torch.unsqueeze(labels, dim=0).unsqueeze(0)
    one_hot = torch.zeros(1, C, n, dtype=torch.long).cuda()         # create black
    target = one_hot.scatter_(1, labels.type(torch.long).data, 1)   # retuqire long type
    return target.type(torch.float32)

def binarys(points, dep):
    coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
    if dep == 0:
        ax = 1
    else:
        ax = 0
    left, right = coord_min[ax], coord_max[ax]
    half_n = int(points.shape[0] / 2.0)
    pidx1, pidx2 = None, None
    while (left + 1e-8 < right):
        mid = (left + right) / 2.0 
        tmp1 = np.where((points[:, ax] <= mid))[0]
        if tmp1.size <= int(half_n*1.1) and tmp1.size >= int(half_n*0.9):
            tmp2 = np.where((points[:, ax] > mid))[0]
            pidx1 = tmp1
            pidx2 = tmp2
            break
        elif tmp1.size < int(half_n*0.9):
            left = mid
        else:
            right = mid
    assert (points.shape[0] == pidx1.size + pidx2.size)
    return pidx1, pidx2

def dfs(points, points_index, th, dep):
    ret = []
    if points.shape[0] <= th:
        ret.append(points_index)
        return ret

    t1, t2 = binarys(points, dep)
    p1, p2 = points[t1, :], points[t2, :]
    i1, i2 = points_index[t1], points_index[t2]
    r1 = dfs(p1, i1, th, dep+1)
    for val in r1:
        ret.append(val)
    r2 = dfs(p2, i2, th, dep+1)
    for val in r2:
        ret.append(val)
    return ret

def split_data(points, rgb, normal, object, semantic, param, offset, th=100000):
    coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
    
    points_idx = np.arange(points.shape[0])
    pidx_list = dfs(points, points_idx, th, 0)
    pts, rgbs, normals, objects, semantics, params, offsets, indexs = [], [], [], [], [], [], [], []
    for val in pidx_list:
        pts.append(points[val, :])
        rgbs.append(rgb[val, :])
        normals.append(normal[val, :])
        objects.append(object[val])
        semantics.append(semantic[val])
        params.append(param[val, :])
        offsets.append(val.shape[0])
        indexs.append(val)
    return pts, rgbs, normals, objects, semantics, params, offsets, indexs
    
def test(test_loader, model, criterion, criterion_re_xyz, criterion_re_label, criterion_re_sp, epoch):
    logger.info('>>>>>>>>>>>>>>>> Start Test >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    loss_re_xyz_meter = AverageMeter()
    loss_re_label_meter = AverageMeter()
    loss_re_sp_meter = AverageMeter()
    # loss_semantic_meter = AverageMeter()
    loss_re_norm_meter = AverageMeter()
    loss_norm_consis_meter = AverageMeter()
    loss_p2sp_contrast_meter = AverageMeter()
    loss_re_param_meter = AverageMeter()

    loss_meter = AverageMeter()
    
    BR_meter = tnt.meter.AverageValueMeter()
    BP_meter = tnt.meter.AverageValueMeter()
    confusion_matrix = metrics.ConfusionMatrix(args['classes'])

    model.eval()
    end = time.time()
    test_start = time.time()
    max_iter = 1 * len(test_loader)
    cnt_room, cnt_sp, cnt_sp_act = 0, 0, 0  # cnt_room: number of rooms, cnt_sp: number of superpoints, cnt_sp_act: number of active superpoints
    cnt_sp_std = 0.
    for i, (filename, coord, rgb, normals, label, semantic, param, offset, edg_source, edg_target, is_transition) in enumerate(test_loader): 
        logger.info('name: {}'.format(filename[0]))
        # fname: file name
        # edg_source: 1
        # edg_target: 1
        # is_transition: 1
        # label: n            torch.IntTensor
        # objects: n                    torch.LongTensor
        # coord: n x 3                  torch.FloatTensor
        # rgb: n x 3                    torch.FloatTensor
        # normals: n x 3                torch.FloatTensor
        # semantic: n x 14              torch.IntTensor
        # param: n x 15                  torch.FloatTensor
        # offset: 1                 
        # coord, rgb, normals, label, semantic, param, offset = coord.cuda(non_blocking=True), rgb.cuda(non_blocking=True), normals.cuda(non_blocking=True), \
        #             label.cuda(non_blocking=True), semantic.cuda(non_blocking=True), param.cuda(non_blocking=True), offset.cuda(non_blocking=True)
        semantic_gt = semantic[:, 1:].argmax(axis=1)
        
        logger.info('xyz: {}'.format(coord.numpy().shape))    # 1 x n x 3
        # th = 500000
        th = 200000      # 1GB显存4500个点
        if coord.size(0) >= th:
            inps, rgbs, normalss, objects, semantics_gt, params, offsets, indexs = split_data(coord.numpy(), rgb.numpy(), normals.numpy(), label.numpy(), semantic_gt.numpy(), param.numpy(), offset, th)
            n = 0
            mov_n = 0   # number of points move
            mov_m = 0   # number of clusters move
            all_c_idx = np.array([], dtype=np.int32)
            # all_c2p_idx = np.zeros((1, coord.size(0), args['near_clusters2point']), dtype=np.int32)   # 1 x n x nc2p
            all_c2p_idx_base = np.zeros((coord.size(0), args['near_clusters2point']), dtype=np.int32)   # 1 x n x nc2p
            # all_output = np.zeros((1, coord.size(0), args['classes']), dtype=np.float32)                  # 1 x n x nclass
            all_rec_xyz = np.zeros((coord.size(0), 3), dtype=np.float32)                               # 1 x n x 3
            all_rec_label = np.zeros((coord.size(0), args['classes']), dtype=np.float32)               # 1 x n x nclass
            all_fea_dist = np.zeros((coord.size(0), args['near_clusters2point']), dtype=np.float32)    # 1 x n x nc2p
            for j in range(len(inps)):
                n = n + inps[j].shape[0]
                s_coord = torch.from_numpy(inps[j]).cuda(non_blocking=True)
                s_rgb = torch.from_numpy(rgbs[j]).cuda(non_blocking=True)
                s_normals = torch.from_numpy(normalss[j]).cuda(non_blocking=True)
                s_semantics_gt = torch.from_numpy(semantics_gt[j]).cuda(non_blocking=True).int()
                s_objects = torch.from_numpy(objects[j]).cuda(non_blocking=True).int()
                s_params = torch.from_numpy(params[j]).cuda(non_blocking=True)
                s_offset = torch.tensor([offsets[j]]).cuda(non_blocking=True).int()
                s_onehot_label = label2one_hot(s_semantics_gt, args['classes']) 

                # logger.info('s_coord: {} {}'.format(s_coord.size(), s_coord.type()))
              
                with torch.no_grad():
                    type_per_point, spout, c_idx, c2p_idx_base, rec_xyz, rec_label, p_fea, sp_pred_lab, sp_pseudo_lab, sp_pseudo_lab_onehot, rec_normal, sp_center_normal, \
                        w_normal_dis, rec_param, contrastive_loss = model([s_coord, s_rgb, s_offset], s_onehot_label, s_semantics_gt, s_objects, s_params, s_normals) # superpoint
                
                c_idx = c_idx.cpu().numpy()                 # 1 x m'         val: 0,1,2,...,n'-1
                c_idx += mov_n
                
                all_c_idx = np.concatenate((all_c_idx, c_idx), axis=0) if all_c_idx.size else c_idx
                # logger.info('c_idx: {}'.format(c_idx.shape, np.max(c_idx)))
                # logger.info('mov_n: {}'.format(mov_n))
            
                c2p_idx_base = c2p_idx_base.cpu().numpy()   # 1 x n' x nc2p  val: 0,1,2,...,m'-1
                c2p_idx_base += mov_m
                all_c2p_idx_base[indexs[j], :] = c2p_idx_base
                # logger.info('c2p_idx_base: {}'.format(c2p_idx_base.shape))
                # logger.info('mov_m: {}'.format(mov_m))
            
                rec_xyz = rec_xyz.detach().cpu().numpy()    # 1 x 3 x n
                # logger.info('rec_xyz: {}'.format(rec_xyz.shape))
                all_rec_xyz[indexs[j], :] = rec_xyz.transpose((1, 0))
            
                rec_label = rec_label.detach().cpu().numpy()    # 1 x nclass x n
                # logger.info('rec_label: {}'.format(rec_label.shape))
                all_rec_label[indexs[j], :] = rec_label.transpose((1, 0))


                fea_dist = spout.detach().cpu().numpy()  # 1 x n' x nc2p
                all_fea_dist[indexs[j], :] = fea_dist
                # logger.info('fea_dist: {}'.format(fea_dist.shape))
            
            
                mov_n += inps[j].shape[0]   # number of points move
                mov_m += c_idx.shape[0]   # number of clusters move
            # logger.info('n: {}'.format(n))
            # logger.info('all_c_idx: {}'.format(all_c_idx.shape))                    # 1 x m

            # logger.info('all_c2p_idx_base: {}'.format(all_c2p_idx_base.shape))      # 1 x n x nc2p
            
            all_rec_xyz = all_rec_xyz.transpose((1, 0))          # 1 x n x 3 -> 1 x 3 x n
            # logger.info('all_rec_xyz: {}'.format(all_rec_xyz.shape))                # 1 x 3 x n
            all_rec_label = all_rec_label.transpose((1, 0))      # 1 x n x nclass -> 1 x nclass x n
            # logger.info('all_rec_label: {}'.format(all_rec_label.shape))            # 1 x nclass x n
            # logger.info('all_fea_dist: {}'.format(all_fea_dist.shape))              # 1 x n x nc2p
            
            semantic_gt = semantic_gt.cuda(non_blocking=True)
            onehot_label = label2one_hot(semantic_gt, args['classes'])

        else:
            coord = coord.cuda(non_blocking=True)
            rgb = rgb.cuda(non_blocking=True)
            offset = offset.cuda(non_blocking=True)
            semantic_gt = semantic_gt.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            param = param.cuda(non_blocking=True)
            normals = normals.cuda(non_blocking=True)
            onehot_label = label2one_hot(semantic_gt, args['classes'])
            
            with torch.no_grad():
                type_per_point, spout, c_idx, c2p_idx_base, rec_xyz, rec_label, p_fea, sp_pred_lab, sp_pseudo_lab, sp_pseudo_lab_onehot, rec_normal, sp_center_normal, \
                 w_normal_dis, rec_param, contrastive_loss = model([coord, rgb, offset], onehot_label, semantic_gt, label, param, normals) # superpoint
            # ---------------- superpoint realted ------------------
            # spout:        b x n x nc2p
            # c_idx:        b x m           in 0,1,2,...,n
            # c2p_idx:      b x n x nc2p    in 0,1,2,...,n
            # c2p_idx_base: b x n x nc2p    in 0,1,2,...,m-1
            # ---------------- semantic related --------------------
            # output:       b x classes x n
            all_c_idx = c_idx.cpu().numpy()
            # all_c2p_idx = c2p_idx.cpu().numpy()
            all_c2p_idx_base = c2p_idx_base.cpu().numpy()
            # all_output = output.detach().cpu().numpy()
            all_rec_xyz = rec_xyz.detach().cpu().numpy()
            all_rec_label = rec_label.detach().cpu().numpy()
            all_fea_dist = spout.detach().cpu().numpy()
            
            coord = coord.cpu()
            semantic_gt = semantic_gt.cpu()

        cnt_sp += all_c_idx.shape[0]
        cnt_room += 1
      
        spout = all_fea_dist
        if semantic_gt.shape[-1] == 1:
            semantic_gt = semantic_gt[:, 0]  # for cls
        # type_loss = args['w_type_loss'] * criterion(type_per_point, semantic_gt)
        re_xyz_loss = args['w_re_xyz_loss'] * criterion_re_xyz(torch.from_numpy(all_rec_xyz), coord.transpose(0,1).contiguous())
        if args['re_label_loss'] == 'cel':
            re_label_loss = args['w_re_label_loss'] * criterion_re_label(torch.from_numpy(all_rec_label).transpose(0,1).contiguous(), semantic_gt.cpu())
        elif args['re_label_loss'] == 'mse':
            re_label_loss = args['w_re_label_loss'] * criterion_re_label(torch.from_numpy(all_rec_label), onehot_label.cpu())

        if args['re_sp_loss'] == 'cel':
            re_sp_loss = args['w_re_sp_loss'] * criterion_re_sp(sp_pred_lab, sp_pseudo_lab)
        elif args['re_sp_loss'] == 'mse':
            re_sp_loss = args['w_re_sp_loss'] * criterion_re_sp(sp_pred_lab, sp_pseudo_lab_onehot)

        # normals = normals.cuda(non_blocking=True)   # 计算loss的时候需要用到
        # param = param.cuda(non_blocking=True)
        # # 重建法线损失以及法线一致性损失
        # re_normal_loss = args['w_re_normal_loss'] * (1 - (1 - w_normal_dis) * torch.sum(normals * rec_normal, dim=1, keepdim=False) / (torch.norm(normals, dim=1, keepdim=False) * torch.norm(rec_normal, dim=1, keepdim=False) + 1e-8)).mean()
        # normal_consistency_loss = args['w_normal_consistency_loss'] * (1 - (1 - w_normal_dis) * torch.sum(sp_center_normal * rec_normal, dim=1, keepdim=False) / (torch.norm(sp_center_normal, dim=1, keepdim=False) * torch.norm(rec_normal, dim=1, keepdim=False) + 1e-8)).mean()

        # # 重建参数损失
        # re_param_loss = args['w_re_param_loss'] * torch.mean(torch.norm(rec_param - param, p=2, dim=1, keepdim=False))

        # # 对比损失
        # contrastive_loss = args['w_contrastive_loss'] * contrastive_loss

        # # 总loss
        # loss = re_xyz_loss + re_label_loss + re_sp_loss + re_normal_loss + normal_consistency_loss + re_param_loss + contrastive_loss

        loss = re_xyz_loss + re_label_loss + re_sp_loss
        # if args['use_semantic_loss']:
        #     semantic_loss = criterion(torch.from_numpy(all_output), gt)
        #     loss = loss + args['w_semantic_loss'] * semantic_loss

        # calculate superpoint metrics
        for bid in range(offset.shape[0]):
            tedg_source = edg_source[bid]
            tedg_target = edg_target[bid]
            tis_transition = is_transition[bid].numpy()
            init_center = all_c_idx
            filename_ = filename[bid]
            if bid == 0:
                txyz = coord[:offset[0]].numpy()
                spout_ = spout[:offset[0]]
                pt_center_index = all_c2p_idx_base[:offset[0]]
                semantic_ = semantic[:offset[0]].numpy()
            else:
                txyz = coord[offset[bid-1]:offset[bid]].numpy()
                spout_ = spout[offset[bid-1]:offset[bid]]
                pt_center_index = all_c2p_idx_base[offset[bid-1]:offset[bid]]
                semantic_ = semantic[offset[bid-1]:offset[bid]].numpy()
            # pred_components, pred_in_component = get_components(init_center, pt_center_index, spout_, getmax=True, trick=False, logger=logger)
            pred_components, pred_in_component, center = get_components(init_center, pt_center_index, spout_, getmax=True, trick=True)
            pred_components = [x[0] for x in pred_components]
            cnt_sp_act += len(pred_components)

            if args.spg_out:
                graph_sp = compute_sp_graph(txyz, 100, pred_in_component, pred_components, semantic_, args.classes)
                spg_file = os.path.join("/data1/fz20/dataset/S3DIS/Stanford3dDataset_v1.2_Aligned_Version/superpoint_graphs_ours", "Area", filename_)
                if not os.path.exists(os.path.dirname(spg_file)):
                    os.makedirs(os.path.dirname(spg_file))
                try:
                    os.remove(spg_file)
                except OSError:
                    pass
                write_spg(spg_file, graph_sp, pred_components, pred_in_component)

            # 可视化
            if args.visual:
                # 超点分割结果
                root_name = os.path.join(args["model_path"], "visual") + '/{}.ply'.format(filename_)
                partition2ply(root_name, txyz, pred_components)

            pred_transition = pred_in_component[tedg_source] != pred_in_component[tedg_target]
            # if args['data_name'] in ['scannet_v1', 'semanticposs']:
            #     full_pred = perfect_prediction_base0(pred_components, pred_in_component, labels[bid, :, :].numpy())
            # else:
            full_pred = perfect_prediction(pred_components, pred_in_component, semantic_)

            # if args['data_name'] in ['scannet_v1', 'semanticposs']:
            #     confusion_matrix.count_predicted_batch(labels[bid, :, :].numpy(), full_pred)
            # else:
            confusion_matrix.count_predicted_batch(semantic_[:, 1:], full_pred)

            if np.sum(tis_transition) > 0:
                BR_meter.add((tis_transition.sum()) * compute_boundary_recall(tis_transition, 
                            relax_edge_binary(pred_transition, tedg_source, 
                            tedg_target, txyz.shape[0], args['BR_tolerance'])),
                            n=tis_transition.sum())
                BP_meter.add((pred_transition.sum()) * compute_boundary_precision(
                            relax_edge_binary(tis_transition, tedg_source, 
                            tedg_target, txyz.shape[0], args['BR_tolerance']), 
                            pred_transition),n=pred_transition.sum())
        
        loss_re_xyz_meter.update(re_xyz_loss.item(), coord.size(0))
        loss_re_label_meter.update(re_label_loss.item(), coord.size(0))
        # if args['use_semantic_loss']:
        #     loss_semantic_meter.update(semantic_loss.item(), coord.size(0))

        loss_meter.update(loss.item(), coord.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        # calculate remain time
        current_iter = i + 1
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        # if args['use_semantic_loss']:
        #     logger.info('Epoch: [{}/{}][{}/{}] '
        #                     'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
        #                     'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
        #                     'Remain {remain_time} '
        #                     'LS_re_xyz {loss_re_xyz_meter.val:.4f} '
        #                     'LS_re_label {loss_re_label_meter.val:.4f} '
        #                     'LS_seg_label {loss_semantic_meter.val:.4f} '
        #                     'Loss {loss_meter.val:.4f}'.format(epoch+1, args["epochs"], i + 1, len(test_loader),
        #                                                       batch_time=batch_time, data_time=data_time,
        #                                                       remain_time=remain_time,
        #                                                       loss_re_xyz_meter=loss_re_xyz_meter,
        #                                                       loss_re_label_meter=loss_re_label_meter,
        #                                                       loss_semantic_meter=loss_semantic_meter,
        #                                                       loss_meter=loss_meter))
        # else:
        logger.info('Epoch: [{}/{}][{}/{}] '
                    'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                    'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                    'Remain {remain_time} '
                    'LS_re_xyz {loss_re_xyz_meter.val:.4f} '
                    'LS_re_label {loss_re_label_meter.val:.4f} '
                    'Loss {loss_meter.val:.4f}'.format(epoch+1, args["epochs"], i + 1, len(test_loader),
                                                        batch_time=batch_time, data_time=data_time,
                                                        remain_time=remain_time,
                                                        loss_re_xyz_meter=loss_re_xyz_meter,
                                                        loss_re_label_meter=loss_re_label_meter,
                                                        loss_meter=loss_meter))

    asa = confusion_matrix.get_overall_accuracy()
    br = BR_meter.value()[0]
    bp = BP_meter.value()[0]
    f1_score = 2 * br * bp / (br + bp + 1e-10)
    test_time = time.time() - test_start
    logger.info('Test result: ASA/BR/BP/F1 {:.4f}/{:.4f}/{:.4f}/{:.4f}'.format(asa, br, bp, f1_score))
    logger.info('cnt_room: {} cnt_sp: {} avg_sp: {}'.format(cnt_room, cnt_sp, 1.*cnt_sp/cnt_room))
    logger.info('cnt_sp_act: {} avg_sp_act: {}'.format(cnt_sp_act, 1.*cnt_sp_act/cnt_room))
    logger.info('Test total time: {:.2f}s'.format(test_time))
    file_result_txt = open(args.save_folder + '/results' + '.txt',"w")
    # file_result_txt = open(args.save_folder + '/results_asa' + '.txt',"w")
    file_result_txt.write("   cnt_room \t cnt_sp \t avg_sp \t cnt_sp_act \t avg_sp_act\n")
    file_result_txt.write("%d \t %d \t %d \t %d \t %d \n" % (cnt_room, cnt_sp, 1.*cnt_sp/cnt_room, cnt_sp_act, 1.*cnt_sp_act/cnt_room) )
    file_result_txt.write("   ASA \t BR \t BP \t F1 \t Test time\n")
    file_result_txt.write("%.4f \t %.4f \t %.4f \t %.4f \t %.2f \n" % (asa, br, bp, f1_score, test_time) )

if __name__ == '__main__':
    main()
