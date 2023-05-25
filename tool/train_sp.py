import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../'))
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import time
import random
import numpy as np
import logging
import argparse
import shutil
import requests

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter

import torchnet as tnt
from util.metrics import *
from util import config
from util.abc import ABC_Dataset
from util.s3dis import S3DIS
from util.common_util import AverageMeter, intersectionAndUnionGPU, find_free_port
from util.data_util import collate_fn, collate_fn_limit
from util import transform as t
from util.logger import get_logger
from util.loss_util import compute_embedding_loss, mean_shift_gpu, compute_iou, sp_mean_shift_gpu, mean_shift
from util.sp_util import get_components, partition2ply, perfect_prediction, relax_edge_binary
from functools import partial
from util.lr import MultiStepWithWarmup, PolyLR


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Point Cloud Primitive Segmentation')
    parser.add_argument('--config', type=str, default='config/abc/abc_sp.yaml', help='config file')
    parser.add_argument('opts', help='see config/abc/abc.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


# def get_logger():
#     logger_name = "main-logger"
#     logger = logging.getLogger(logger_name)
#     logger.setLevel(logging.INFO)
#     handler = logging.StreamHandler()
#     fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
#     handler.setFormatter(logging.Formatter(fmt))
#     logger.addHandler(handler)
#     return logger


def worker_init_fn(worker_id):
    random.seed(args.manual_seed + worker_id)


def main_process():
    return not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)


def main():
    args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    # headers = {"Authorization": "eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1aWQiOjExNjc4OCwidXVpZCI6ImQzZjBmOGIyLWJmNWMtNDkyMy1hMzMyLTEyN2ViNTg4ZDEyMCIsImlzX2FkbWluIjpmYWxzZSwiaXNfc3VwZXJfYWRtaW4iOmZhbHNlLCJzdWJfbmFtZSI6IiIsInRlbmFudCI6ImF1dG9kbCIsInVwayI6IiJ9.ma-DgwI8MuctNwGWyoVoIVfr7r0Gt64nwJA_U4FToy2lg4ueMhWPlybP0UxP-yKw5_KpfNil4EcWe7t5Wc5Irw"}
    # resp = requests.post("https://www.autodl.com/api/v1/wechat/message/send",
    #                  json={
    #                      "title": "A100: begin training",
    #                      "name": "Your model starts training",
    #                      "content": "Using GPU # {}, the save path is {}".format(args.train_gpu, args.save_path)
    #                  }, headers = headers)

    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False

    # if args.data_name == 's3dis':
    #     S3DIS(split='train', data_root=args.data_root, test_area=args.test_area)
    #     S3DIS(split='val', data_root=args.data_root, test_area=args.test_area)
    # else:
    #     raise NotImplementedError()
    if args.multiprocessing_distributed:
        port = find_free_port()
        args.dist_url = f"tcp://localhost:{port}"
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, argss):
    global args, best_iou, best_f1_score
    args, best_iou, best_f1_score = argss, 0, 0
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    from model.pointtransformer.pointtransformer_seg import BoundaryNet as BoundaryModel
    if args.arch == 'pointtransformer_seg_repro':
        from model.pointtransformer.pointtransformer_seg import pointtransformer_seg_repro as Model
    elif args.arch == 'pointtransformer_primitive_seg_repro':
        from model.pointtransformer.pointtransformer_seg import PointTransformer_PrimSeg as Model
    elif args.arch == 'boundarytransformer_primitive_seg_repro':
        from model.pointtransformer.pointtransformer_seg import BoundaryTransformer_PrimSeg as Model
    elif args.arch == 'pointtransformer_Unit_seg_repro':
        from model.pointtransformer.pointtransformer_seg import pointtransformer_Unit_seg_repro as Model
    elif args.arch == 'boundarypointtransformer_Unit_seg_repro':
        from model.pointtransformer.pointtransformer_seg import boundarypointtransformer_Unit_seg_repro as Model
    elif args.arch == 'superpoint_net':
        from model.superpoint.superpoint_net import superpoint_seg_repro as Model
    else:
        raise Exception('architecture {} not supported yet'.format(args.arch))
    # model = Model(c=args.fea_dim, k=args.classes)
    # boundarymodel = BoundaryModel(c=args.fea_dim, k=args.classes, args=args)
    model = Model(c=args.fea_dim, k=args.classes, args=args)

    # if args.sync_bn:
    #    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda()
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

    # optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.epochs*0.6), int(args.epochs*0.8)], gamma=0.1)

    # set optimizer
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'AdamW':     # Adamw 即 Adam + weight decate ,效果与 Adam + L2正则化相同,但是计算效率更高
        # transformer_lr_scale = args.get("transformer_lr_scale", 0.1)
        # param_dicts = [
        #     {"params": [p for n, p in model.named_parameters() if "blocks" not in n and p.requires_grad]},
        #     {
        #         "params": [p for n, p in model.named_parameters() if "blocks" in n and p.requires_grad],
        #         "lr": args.base_lr * transformer_lr_scale,
        #     },
        # ]
        # optimizer = torch.optim.AdamW(param_dicts, lr=args.base_lr, weight_decay=args.weight_decay)
        # optimizer = torch.optim.AdamW([{"params": boundarymodel.parameters()}, {"params": model.parameters()}], lr=args.base_lr, weight_decay=args.weight_decay)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)


    if main_process():
        global logger, writer
        logger = get_logger(args.save_path)
        writer = SummaryWriter(args.save_path)
        logger.info(args)
        logger.info("=> creating model ...")
        logger.info("Classes: {}".format(args.classes))
        # logger.info(boundarymodel)
        logger.info(model)
        # logger.info('#Model parameters: {}'.format(sum([x.nelement() for x in boundarymodel.parameters()]) + sum([x.nelement() for x in model.parameters()])))
        total = sum([param.nelement() for param in model.parameters()])
        logger.info("Number of params: %.2fM" % (total/1e6))
        if args.get("max_grad_norm", None):
            logger.info("args.max_grad_norm = {}".format(args.max_grad_norm))

    if args.distributed:
        torch.cuda.set_device(gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.batch_size_val = int(args.batch_size_val / ngpus_per_node)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        if args.sync_bn:
            if main_process():
                logger.info("use SyncBN")
            # boundarymodel = torch.nn.SyncBatchNorm.convert_sync_batchnorm(boundarymodel).cuda()
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
        # boundarymodel = torch.nn.parallel.DistributedDataParallel(
        #     boundarymodel,
        #     device_ids=[gpu],
        #     find_unused_parameters=True
        # )
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[gpu],
            find_unused_parameters=True
        )

    else:
        # boundarymodel = torch.nn.DataParallel(boundarymodel.cuda())
        model = torch.nn.DataParallel(model.cuda())

    if args.weight:
        if os.path.isfile(args.weight):
            if main_process():
                logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight)
            # boundarymodel.load_state_dict(checkpoint['boundary_state_dict'])
            model.load_state_dict(checkpoint['state_dict'])
            if main_process():
                logger.info("=> loaded weight '{}'".format(args.weight))
        else:
            logger.info("=> no weight found at '{}'".format(args.weight))

    if args.resume:
        if os.path.isfile(args.resume):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
            args.start_epoch = checkpoint['epoch']
            # boundarymodel.load_state_dict(checkpoint['boundary_state_dict'], strict=True)
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler_state_dict = checkpoint['scheduler']
            #best_iou = 40.0
            # best_iou = checkpoint['best_iou']
            if main_process():
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            if main_process():
                logger.info("=> no checkpoint found at '{}'".format(args.resume))

    # train_transform = t.Compose([t.RandomScale([0.9, 1.1]), t.ChromaticAutoContrast(), t.ChromaticTranslation(), t.ChromaticJitter(), t.HueSaturationTranslation()])
    # train_data = S3DIS(split='train', data_root=args.data_root, test_area=args.test_area, voxel_size=args.voxel_size, voxel_max=args.voxel_max, transform=train_transform, shuffle_index=True, loop=args.loop)
    if args.data_name == 'abc':
        train_data = ABC_Dataset(split='train', data_root=args.data_root, voxel_size=args.voxel_size, voxel_max=args.voxel_max, shuffle_index=True, loop=args.train_loop)
    else:
        raise ValueError("The dataset {} is not supported.".format(args.data_name))

    if main_process():
            logger.info("train_data samples: '{}'".format(len(train_data)))
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True, collate_fn=collate_fn)
    # train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.workers, \
    #     pin_memory=True, sampler=train_sampler, drop_last=True, collate_fn=partial(collate_fn_limit, max_batch_points=args.max_batch_points, logger=logger if main_process() else None))

    # val_loader = None
    if args.evaluate:
        # val_transform = None
        # val_data = S3DIS(split='val', data_root=args.data_root, test_area=args.test_area, voxel_size=args.voxel_size, voxel_max=800000, transform=val_transform)
        if args.data_name == 'abc':
            val_data = ABC_Dataset(split='val', data_root=args.data_root, voxel_size=args.voxel_size, voxel_max=800000, loop=args.val_loop)
        else:
            raise ValueError("The dataset {} is not supported.".format(args.data_name))
        if args.distributed:
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
        else:
            val_sampler = None
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False, num_workers=args.workers, pin_memory=True, sampler=val_sampler, collate_fn=collate_fn)

    # set scheduler
    if args.scheduler == "MultiStepWithWarmup":
        assert args.scheduler_update == 'step'
        if main_process():
            logger.info("scheduler: MultiStepWithWarmup. scheduler_update: {}".format(args.scheduler_update))
        iter_per_epoch = len(train_loader)
        milestones = [int(args.epochs*0.6) * iter_per_epoch, int(args.epochs*0.8) * iter_per_epoch]
        scheduler = MultiStepWithWarmup(optimizer, milestones=milestones, gamma=0.1, warmup=args.warmup, \
            warmup_iters=args.warmup_iters, warmup_ratio=args.warmup_ratio)
    elif args.scheduler == 'MultiStep':
        assert args.scheduler_update == 'epoch'
        milestones = [int(x) for x in args.milestones.split(",")] if hasattr(args, "milestones") else [int(args.epochs*0.4), int(args.epochs*0.8)]
        gamma = args.gamma if hasattr(args, 'gamma') else 0.1
        if main_process():
            logger.info("scheduler: MultiStep. scheduler_update: {}. milestones: {}, gamma: {}".format(args.scheduler_update, milestones, gamma))
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)     # 当前epoch数满足设定值时，调整学习率
    elif args.scheduler == 'Poly':
        if main_process():
            logger.info("scheduler: Poly. scheduler_update: {}".format(args.scheduler_update))
        if args.scheduler_update == 'epoch':
            scheduler = PolyLR(optimizer, max_iter=args.epochs, power=args.power)
        elif args.scheduler_update == 'step':
            iter_per_epoch = len(train_loader)
            scheduler = PolyLR(optimizer, max_iter=args.epochs*iter_per_epoch, power=args.power)
        else:
            raise ValueError("No such scheduler update {}".format(args.scheduler_update))
    else:
        raise ValueError("No such scheduler {}".format(args.scheduler))

    if args.resume and os.path.isfile(args.resume):
        scheduler.load_state_dict(scheduler_state_dict)
        print("resume scheduler")

    ###################
    # start training #
    ###################

    if args.use_amp:    # 自动混合精度训练 —— 节省显存并加快推理速度
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    # 测试与可视化
    if args.evaluate:
        with torch.no_grad():
            loss_val, br, bp, f1 = validate(val_loader, model, criterion, criterion_re_xyz, criterion_re_label, criterion_re_sp)
        
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        if main_process():
            logger.info("lr: {}".format(scheduler.get_last_lr()))

        # loss_train, mIoU_train, mAcc_train, allAcc_train = train(train_loader, model, criterion, optimizer, epoch)
        loss_train, loss_normal, loss_param, loss_contrast, loss_feat, loss_type, f1_score = train(train_loader, model, criterion, criterion_re_xyz, criterion_re_label, criterion_re_sp, optimizer, epoch, scaler, scheduler)
        if args.scheduler_update == 'epoch':
            scheduler.step()
        epoch_log = epoch + 1
        is_best = False
        if main_process():
            writer.add_scalar('feat_loss_train', loss_feat, epoch_log)
            writer.add_scalar('type_loss_train', loss_type, epoch_log)
            writer.add_scalar('normal_loss_train', loss_normal, epoch_log)
            writer.add_scalar('param_loss_train', loss_param, epoch_log)
            writer.add_scalar('contrast_loss_train', loss_contrast, epoch_log)
            writer.add_scalar('loss_train', loss_train, epoch_log)
            writer.add_scalar('f1_score', f1_score, epoch_log)
            # writer.add_scalar('mIoU_train', mIoU_train, epoch_log)
            # writer.add_scalar('mAcc_train', mAcc_train, epoch_log)
            # writer.add_scalar('allAcc_train', allAcc_train, epoch_log)
            is_best = f1_score > best_f1_score
            best_f1_score = max(f1_score, best_f1_score)

        # is_best = False
        # if args.evaluate and (epoch_log % args.eval_freq == 0):
        #     if args.data_name == 'shapenet':
        #         raise NotImplementedError()
        #     else:
        #         loss_val, mIoU_val, mAcc_val, allAcc_val = validate(val_loader, model, criterion, criterion_re_xyz, criterion_re_label, criterion_re_sp)
        #     # s_miou, p_miou, feat_loss_val, type_loss_val, boundary_loss_val = validate(val_loader, model, criterion)

        #     if main_process():
        #         # writer.add_scalar('feat_loss_val', feat_loss_val, epoch_log)
        #         # writer.add_scalar('type_loss_val', type_loss_val, epoch_log)
        #         # writer.add_scalar('boundary_loss_val', boundary_loss_val, epoch_log)
        #         # writer.add_scalar('s_miou', s_miou, epoch_log)
        #         # writer.add_scalar('p_miou', p_miou, epoch_log)
        #         writer.add_scalar('loss_val', loss_val, epoch_log)
        #         writer.add_scalar('mIoU_val', mIoU_val, epoch_log)
        #         writer.add_scalar('mAcc_val', mAcc_val, epoch_log)
        #         writer.add_scalar('allAcc_val', allAcc_val, epoch_log)
        #         # is_best = s_miou > best_iou
        #         # best_iou = max(best_iou, s_miou)

        # if (epoch_log % args.save_freq == 0) and main_process():
        #     if not os.path.exists(args.save_path + "/model/"):
        #         os.makedirs(args.save_path + "/model/")
        #     filename = args.save_path + '/model/model_last.pth'
        #     logger.info('Saving checkpoint to: ' + filename)
        #     torch.save({'epoch': epoch_log, 'boundary_state_dict': boundarymodel.state_dict(), 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
        #                 'scheduler': scheduler.state_dict(), 'best_iou': best_iou, 'is_best': is_best}, filename)
        #     if is_best:
        #         logger.info('Best validation mIoU updated to: {:.4f}'.format(best_iou))
        #         shutil.copyfile(filename, args.save_path + '/model/model_best.pth')
        if (epoch_log % args.save_freq == 0) and main_process():
            if not os.path.exists(args.save_path + "/model/"):
                os.makedirs(args.save_path + "/model/")
            # filename = os.path.join(args["save_path"], 'model/train_epoch_{}.pth'.format(str(epoch_log)))
            filename = args.save_path + '/model/model_last.pth'
            logger.info('Saving checkpoint to: ' + filename)
            if scheduler is not None:
                torch.save({'epoch': epoch_log, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}, filename)
            else:
                torch.save({'epoch': epoch_log, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, filename)

            if is_best:
                logger.info('Best validation f1_score updated to: {:.4f}'.format(best_f1_score))
                shutil.copyfile(filename, args.save_path + '/model/model_best.pth')

    if main_process():
        writer.close()
        logger.info('==>Training done!')
        headers = {"Authorization": "eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1aWQiOjExNjc4OCwidXVpZCI6ImQzZjBmOGIyLWJmNWMtNDkyMy1hMzMyLTEyN2ViNTg4ZDEyMCIsImlzX2FkbWluIjpmYWxzZSwiaXNfc3VwZXJfYWRtaW4iOmZhbHNlLCJzdWJfbmFtZSI6IiIsInRlbmFudCI6ImF1dG9kbCIsInVwayI6IiJ9.ma-DgwI8MuctNwGWyoVoIVfr7r0Gt64nwJA_U4FToy2lg4ueMhWPlybP0UxP-yKw5_KpfNil4EcWe7t5Wc5Irw"}
        resp = requests.post("https://www.autodl.com/api/v1/wechat/message/send",
            json={
                "title": "A100: The training is over",
                "name": "End of use of GPU #{}".format(args.train_gpu),
                "content": "The training is over, please check the result"
            }, headers = headers)
        

def label2one_hot(labels, C=10):
    n = labels.size(0)
    labels = torch.unsqueeze(labels, dim=0).unsqueeze(0)
    one_hot = torch.zeros(1, C, n, dtype=torch.long).cuda()         # create black
    target = one_hot.scatter_(1, labels.type(torch.long).data, 1)   # retuqire long type
    return target.type(torch.float32)


def train(train_loader, model, criterion, criterion_re_xyz, criterion_re_label, criterion_re_sp, optimizer, epoch, scaler, scheduler):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_re_xyz_meter = AverageMeter()
    loss_re_label_meter = AverageMeter()
    loss_re_sp_meter = AverageMeter()
    loss_norm_meter = AverageMeter()
    loss_sp_contrast_meter = AverageMeter()
    loss_p2sp_contrast_meter = AverageMeter()
    loss_param_meter = AverageMeter()
    loss_meter = AverageMeter()
    # intersection_meter = AverageMeter()
    # union_meter = AverageMeter()
    # target_meter = AverageMeter()
    feat_loss_meter = AverageMeter()
    type_loss_meter = AverageMeter()
    # boundary_loss_meter = AverageMeter()

    BR_meter = tnt.meter.AverageValueMeter()    # boundary recall
    BP_meter = tnt.meter.AverageValueMeter()    # boundary precision

    # boundarymodel.train()
    model.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)
    print('$'*10)

    for i, (coord, normals, boundary, label, semantic, param, offset, edges, filename, edg_source, edg_target, is_transition) in enumerate(train_loader):  # (n, 3), (n, c), (n), (b)
        data_time.update(time.time() - end)
        # coord, feat, target, offset = coord.cuda(non_blocking=True), feat.cuda(non_blocking=True), target.cuda(non_blocking=True), offset.cuda(non_blocking=True)
        coord, normals, label, semantic, param, offset = coord.cuda(non_blocking=True), normals.cuda(non_blocking=True), \
                    label.cuda(non_blocking=True), semantic.cuda(non_blocking=True), param.cuda(non_blocking=True), offset.cuda(non_blocking=True)
        # coord, normals, boundary, label, semantic, param, offset, edges = coord.cuda(non_blocking=True), normals.cuda(non_blocking=True), boundary.cuda(non_blocking=True), \
        #             label.cuda(non_blocking=True), semantic.cuda(non_blocking=True), param.cuda(non_blocking=True), offset.cuda(non_blocking=True), edges.cuda(non_blocking=True)


        # if args.concat_xyz:
        #     feat = torch.cat([normals, coord], 1)

        use_amp = args.use_amp
        with torch.cuda.amp.autocast(enabled=use_amp):
            # boundary_pred = boundarymodel([coord, normals, offset])
            # softmax = torch.nn.Softmax(dim=1)
            # boundary_pred_ = softmax(boundary_pred)
            # boundary_pred_ = (boundary_pred_[:,1] > 0.5).int()
            onehot_label = label2one_hot(semantic, args['classes'])
            # primitive_embedding, type_per_point = model([coord, normals, offset], edges, boundary_pred_)
            spout, c_idx, c2p_idx_base, type_per_point, primitive_embedding, rec_xyz, rec_label, p_fea, sp_pred_lab, sp_pseudo_lab, sp_pseudo_lab_onehot, normal_loss, sp_center_contrast_loss, p2sp_contrast_loss, param_loss = model([coord, normals, offset], onehot_label, semantic, label, param) # superpoint
            # assert type_per_point.shape[1] == args.classes
            if semantic.shape[-1] == 1:
                semantic = semantic[:, 0]  # for cls
            # loss = criterion(output, target)
            feat_loss, pull_loss, push_loss = compute_embedding_loss(primitive_embedding, label, offset)
            type_loss = criterion(type_per_point, semantic)
            # boundary_loss = criterion(boundary_pred, boundary)
            # loss = feat_loss + type_loss + boundary_loss
            # re_xyz_loss = args['w_re_xyz_loss'] * criterion_re_xyz(rec_xyz.squeeze(0), coord.transpose(0,1).contiguous())    # compact loss
            re_xyz_loss = args['w_re_xyz_loss'] * criterion_re_xyz(rec_xyz, coord.transpose(0,1).contiguous())    # compact loss
            if args['re_label_loss'] == 'cel':
                # re_label_loss = args['w_re_label_loss'] * criterion_re_label(rec_label, semantic.unsqueeze(0)) # point label loss
                re_label_loss = args['w_re_label_loss'] * criterion_re_label(rec_label.transpose(0,1).contiguous(), semantic) # point label loss
            elif args['re_label_loss'] == 'mse':
                re_label_loss = args['w_re_label_loss'] * criterion_re_label(rec_label, onehot_label)

            if args['re_sp_loss'] == 'cel':
                re_sp_loss = args['w_re_sp_loss'] * criterion_re_sp(sp_pred_lab, sp_pseudo_lab) # superpoint label loss
            elif args['re_sp_loss'] == 'mse':
                re_sp_loss = args['w_re_sp_loss'] * criterion_re_sp(sp_pred_lab, sp_pseudo_lab_onehot)

            # loss = re_xyz_loss + re_label_loss + re_sp_loss + normal_loss + sp_center_contrast_loss
            loss = re_xyz_loss + normal_loss + param_loss + p2sp_contrast_loss + feat_loss + type_loss
            # loss = re_xyz_loss + re_label_loss + re_sp_loss + normal_loss + p2sp_contrast_loss
            # loss = re_xyz_loss + normal_loss
            # loss = normal_loss

        # calculate superpoint metrics
        for bid in range(offset.shape[0]):
            tedg_source = edg_source[bid]
            tedg_target = edg_target[bid]
            tis_transition = is_transition[bid].numpy()
            init_center = c_idx.cpu().numpy()
            if bid == 0:
                txyz = coord[:offset[0]].cpu().numpy()
                spout_ = spout[:offset[0]].detach().cpu().numpy()
                # init_center = c_idx[:sp_offset[0]].cpu().numpy()
                pt_center_index = c2p_idx_base[:offset[0]].cpu().numpy()
                # label_ = label[:offset[0]].cpu().numpy()
                p_fea_ = p_fea[:offset[0]]
                primitive_embedding_ = primitive_embedding[:offset[0]]
            else:
                txyz = coord[offset[bid-1]:offset[bid]].cpu().numpy()
                spout_ = spout[offset[bid-1]:offset[bid]].detach().cpu().numpy()
                # init_center = c_idx[sp_offset[bid-1]:sp_offset[bid]].cpu().numpy()
                pt_center_index = c2p_idx_base[offset[bid-1]:offset[bid]].cpu().numpy()
                # label_ = label[offset[bid-1]:offset[bid]].cpu().numpy()
                p_fea_ = p_fea[offset[bid-1]:offset[bid]]
                primitive_embedding_ = primitive_embedding[offset[bid-1]:offset[bid]]
            pred_components, pred_in_component, center = get_components(init_center, pt_center_index, spout_, getmax=True)
            # pred_components = [x[0] for x in pred_components]
            # partition2ply('./sp_test.ply', txyz, pred_components)
            # # 超点均值聚类
            # pred_sp = torch.from_numpy(pred_in_component).to(primitive_embedding_.device)
            # p2sp_matrix = label2one_hot(pred_sp, pred_sp.max()+1).squeeze()
            # sp_cnt = torch.sum(p2sp_matrix, dim=1)
            # sp_fea = torch.matmul(p2sp_matrix, primitive_embedding_) / sp_cnt.unsqueeze(1)
            # cluster_pred = sp_mean_shift_gpu(sp_fea, bandwidth = 2)
            # for i in range(cluster_pred.shape[0]):
            #     pred_in_component[pred_in_component == i] = cluster_pred[i]
            # coms=np.unique(pred_in_component)
            # components = []
            # for i in range(len(coms)):
            #     te=np.where(pred_in_component==coms[i])
            #     components.append(te)
            # partition2ply('./prim_test.ply', txyz, components)

            pred_transition = pred_in_component[tedg_source] != pred_in_component[tedg_target]
            
            # full_pred = perfect_prediction(pred_components, pred_in_component, label_)

            if np.sum(tis_transition) > 0:
                BR_meter.add((tis_transition.sum()) * compute_boundary_recall(tis_transition, 
                            relax_edge_binary(pred_transition, tedg_source, 
                            tedg_target, txyz.shape[0], args['BR_tolerance'])),
                            n=tis_transition.sum())
                BP_meter.add((pred_transition.sum()) * compute_boundary_precision(
                            relax_edge_binary(tis_transition, tedg_source, 
                            tedg_target, txyz.shape[0], args['BR_tolerance']), 
                            pred_transition),n=pred_transition.sum())


        # for j in range(offset.shape[0]):
        #     init_center = c_idx[j, :].cpu().numpy()
        #     filename_ = filename[j]
        #     if j == 0:
        #         xyz = coord[:offset[0]].cpu().numpy()
        #         spout_ = spout[:offset[0]].detach().cpu().numpy()
        #         pt_center_index = c2p_idx_base[:, :offset[0]].squeeze(0).cpu().numpy()
        #     else:
        #         xyz = coord[offset[j-1]:offset[j]].cpu().numpy()
        #         spout_ = spout[offset[j-1]:offset[j]].detach().cpu().numpy()
        #         pt_center_index = c2p_idx_base[:, offset[j-1]:offset[j]].squeeze(0).cpu().numpy()
        #     pred_components, pred_in_component, center = get_components(init_center, pt_center_index, spout_, getmax=True)
        #     # time_tag = time.strftime("%Y%m%d-%H%M%S")
        #     root_name = '/data/fz20/project/point-transformer-boundary/visual/sp_v2_vis/{}.ply'.format(filename_)
        #     partition2ply(root_name, xyz, pred_components)

        # for j in range(offset.shape[0]):
        #     init_center = c_idx.cpu().numpy()
        #     filename_ = filename[j]
        #     if j == 0:
        #         # init_center = c_idx[:sp_offset[0]].cpu().numpy()
        #         xyz = coord[:offset[0]].cpu().numpy()
        #         spout_ = spout[:offset[0]].detach().cpu().numpy()
        #         pt_center_index = c2p_idx_base[:offset[0]].cpu().numpy()
        #     else:
        #         # init_center = c_idx[sp_offset[j-1]:sp_offset[j]].cpu().numpy()
        #         xyz = coord[offset[j-1]:offset[j]].cpu().numpy()
        #         spout_ = spout[offset[j-1]:offset[j]].detach().cpu().numpy()
        #         pt_center_index = c2p_idx_base[offset[j-1]:offset[j]].cpu().numpy()
        #     pred_components, pred_in_component, center = get_components(init_center, pt_center_index, spout_, getmax=True)
        #     # time_tag = time.strftime("%Y%m%d-%H%M%S")
        #     root_name = '/data/fz20/project/point-transformer-boundary/visual/sp_v2_only_normconsisloss/{}.ply'.format(filename_)
        #     partition2ply(root_name, xyz, pred_components)

            
        optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if args.scheduler_update == 'step':
            scheduler.step()

        # output = output.max(1)[1]
        n = coord.size(0)
        # if args.multiprocessing_distributed:
        #     loss *= n
        #     count = semantic.new_tensor([n], dtype=torch.long)
        #     dist.all_reduce(loss), dist.all_reduce(count)
        #     n = count.item()
        #     loss /= n
        # intersection, union, target = intersectionAndUnionGPU(output, semantic, args.classes, args.ignore_label)
        # if args.multiprocessing_distributed:
        #     dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
        # intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        # intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        # accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        feat_loss_meter.update(feat_loss.item(), n)
        type_loss_meter.update(type_loss.item(), n)
        loss_re_xyz_meter.update(re_xyz_loss.item(), n)
        loss_re_label_meter.update(re_label_loss.item(), n)
        loss_re_sp_meter.update(re_sp_loss.item(), n)
        loss_norm_meter.update(normal_loss.item(), n)
        loss_param_meter.update(param_loss.item(), n)
        loss_sp_contrast_meter.update(sp_center_contrast_loss.item(), n)
        loss_p2sp_contrast_meter.update(p2sp_contrast_loss.item(), n)
        loss_meter.update(loss.item(), n)
        
        # # All Reduce loss
        if args.multiprocessing_distributed:
            dist.all_reduce(re_xyz_loss.div_(torch.cuda.device_count()))
            dist.all_reduce(re_label_loss.div_(torch.cuda.device_count()))
            dist.all_reduce(re_sp_loss.div_(torch.cuda.device_count()))
            dist.all_reduce(normal_loss.div_(torch.cuda.device_count()))
            dist.all_reduce(param_loss.div_(torch.cuda.device_count()))
            dist.all_reduce(sp_center_contrast_loss.div_(torch.cuda.device_count()))
            dist.all_reduce(p2sp_contrast_loss.div_(torch.cuda.device_count()))
        #     dist.all_reduce(feat_loss.div_(torch.cuda.device_count()))
        #     dist.all_reduce(type_loss.div_(torch.cuda.device_count()))
        #     dist.all_reduce(boundary_loss.div_(torch.cuda.device_count()))
        # feat_loss_, type_loss_, boundary_loss_ = feat_loss.data.cpu().numpy(), type_loss.data.cpu().numpy(), boundary_loss.data.cpu().numpy()
        # feat_loss_meter.update(feat_loss_.item())
        # type_loss_meter.update(type_loss_.item())
        # boundary_loss_meter.update(boundary_loss_.item())
        batch_time.update(time.time() - end)
        end = time.time()

        # calculate remain time
        current_iter = epoch * len(train_loader) + i + 1
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        # if (i + 1) % args.print_freq == 0 and main_process():
        #     lr = scheduler.get_last_lr()
        #     if isinstance(lr, list):
        #         lr = [round(x, 8) for x in lr]
        #     elif isinstance(lr, float):
        #         lr = round(lr, 8)
        #     logger.info('Epoch: [{}/{}][{}/{}] '
        #                 'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
        #                 'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
        #                 'Remain {remain_time} '
        #                 # 'Loss {loss_meter.val:.4f} '
        #                 'Feat_Loss {feat_loss_meter.val:.4f} '
        #                 'Type_Loss {type_loss_meter.val:.4f} '
        #                 'Boundary_Loss {boundary_loss_meter.val:.4f} '
        #                 'Lr: {lr}.'.format(epoch+1, args.epochs, i + 1, len(train_loader),
        #                                                   batch_time=batch_time, data_time=data_time,
        #                                                   remain_time=remain_time,
        #                                                   feat_loss_meter=feat_loss_meter,
        #                                                   type_loss_meter=type_loss_meter,
        #                                                   boundary_loss_meter=boundary_loss_meter,
        #                                                   lr=lr))

        if (i + 1) % args.print_freq == 0 and main_process():
            lr = scheduler.get_last_lr()
            if isinstance(lr, list):
                lr = [round(x, 8) for x in lr]
            elif isinstance(lr, float):
                lr = round(lr, 8)
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'LS_re_xyz {loss_re_xyz_meter.val:.4f} '
                        'LS_re_label {loss_re_label_meter.val:.4f} '
                        'LS_re_sp {loss_re_sp_meter.val:.4f} '
                        'LS_norm {loss_norm_meter.val:.4f} '
                        'LS_param {loss_param_meter.val:.4f} '
                        'LS_sp_contrast {loss_sp_contrast_meter.val:.4f} '
                        'LS_p2sp_contrast {loss_p2sp_contrast_meter.val:.4f} '
                        'LS_feat {feat_loss_meter.val:.4f} '
                        'LS_type {type_loss_meter.val:.4f} '
                        'Loss {loss_meter.val:.4f} '
                        'lr {lr} '.format(epoch+1, args["epochs"], i + 1, len(train_loader),
                                                          batch_time=batch_time, data_time=data_time,
                                                          remain_time=remain_time,
                                                          loss_re_xyz_meter=loss_re_xyz_meter,
                                                          loss_re_label_meter=loss_re_label_meter,
                                                          loss_re_sp_meter=loss_re_sp_meter,
                                                          loss_norm_meter=loss_norm_meter,
                                                          loss_param_meter=loss_param_meter,
                                                          loss_sp_contrast_meter=loss_sp_contrast_meter,
                                                          loss_p2sp_contrast_meter=loss_p2sp_contrast_meter,
                                                          feat_loss_meter=feat_loss_meter,
                                                          type_loss_meter=type_loss_meter,
                                                          loss_meter=loss_meter,
                                                          lr=lr))

        if main_process():
            writer.add_scalar('feat_loss_train_batch', feat_loss_meter.val, current_iter)
            writer.add_scalar('type_loss_train_batch', type_loss_meter.val, current_iter)
            writer.add_scalar('norm_loss_train_batch', loss_norm_meter.val, current_iter)
            writer.add_scalar('param_loss_train_batch', loss_param_meter.val, current_iter)
            writer.add_scalar('sp_contrast_loss_train_batch', loss_sp_contrast_meter.val, current_iter)
            writer.add_scalar('p2sp_contrast_loss_train_batch', loss_p2sp_contrast_meter.val, current_iter)
            writer.add_scalar('loss_train_batch', loss_meter.val, current_iter)
            # writer.add_scalar('mIoU_train_batch', np.mean(intersection / (union + 1e-10)), current_iter)
            # writer.add_scalar('mAcc_train_batch', np.mean(intersection / (target + 1e-10)), current_iter)
            # writer.add_scalar('allAcc_train_batch', accuracy, current_iter)

            # writer.add_scalar('feat_loss_train_batch', feat_loss_meter.val, current_iter)
            # writer.add_scalar('type_loss_train_batch', type_loss_meter.val, current_iter)
            # writer.add_scalar('boundary_loss_train_batch', boundary_loss_meter.val, current_iter)
            # writer.add_scalar('loss_train_batch', loss_meter.val, current_iter)
            # writer.add_scalar('mIoU_train_batch', np.mean(intersection / (union + 1e-10)), current_iter)
            # writer.add_scalar('mAcc_train_batch', np.mean(intersection / (target + 1e-10)), current_iter)
            # writer.add_scalar('allAcc_train_batch', accuracy, current_iter)

    br = BR_meter.value()[0]
    bp = BP_meter.value()[0]
    f1_score = 2 * br * bp / (br + bp + 1e-10)
    if main_process():
        logger.info('Train result at epoch [{}/{}]: BR/BP/F1-score {:.4f}/{:.4f}/{:.4f}'.format(
                    epoch+1, args['epochs'], br, bp, f1_score))

    # iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    # accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    # mIoU = np.mean(iou_class)
    # mAcc = np.mean(accuracy_class)
    # allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    # if main_process():
    #     logger.info('Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(epoch+1, args.epochs, mIoU, mAcc, allAcc))
    return loss_meter.avg, loss_norm_meter.avg, loss_param_meter.avg, loss_p2sp_contrast_meter.avg, feat_loss_meter.avg, type_loss_meter.avg, f1_score

    # return feat_loss_meter.avg, type_loss_meter.avg, boundary_loss_meter.avg


def validate(val_loader, model, criterion, criterion_re_xyz, criterion_re_label, criterion_re_sp):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_re_xyz_meter = AverageMeter()
    loss_re_label_meter = AverageMeter()
    loss_re_sp_meter = AverageMeter()
    loss_norm_meter = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    # feat_loss_meter = AverageMeter()
    # type_loss_meter = AverageMeter()
    # boundary_loss_meter = AverageMeter()
    # s_iou_meter = AverageMeter()
    # type_iou_meter = AverageMeter()

    BR_meter = tnt.meter.AverageValueMeter()    # boundary recall
    BP_meter = tnt.meter.AverageValueMeter()    # boundary precision

    torch.cuda.empty_cache()

    # boundarymodel.eval()
    model.eval()
    end = time.time()
    for i, (coord, normals, boundary, label, semantic, param, offset, edges, filename, edg_source, edg_target, is_transition) in enumerate(val_loader):
        data_time.update(time.time() - end)
        # coord, feat, target, offset = coord.cuda(non_blocking=True), feat.cuda(non_blocking=True), target.cuda(non_blocking=True), offset.cuda(non_blocking=True)
        coord, normals, boundary, label, semantic, param, offset, edges = coord.cuda(non_blocking=True), normals.cuda(non_blocking=True), boundary.cuda(non_blocking=True), \
                    label.cuda(non_blocking=True), semantic.cuda(non_blocking=True), param.cuda(non_blocking=True), offset.cuda(non_blocking=True), edges.cuda(non_blocking=True)

        if semantic.shape[-1] == 1:
            semantic = semantic[:, 0]  # for cls
        
        # if args.concat_xyz:
        #     feat = torch.cat([normals, coord], 1)
        
        with torch.no_grad():
            # boundary_pred = boundarymodel([coord, normals, offset])
            # softmax = torch.nn.Softmax(dim=1)
            # boundary_pred_ = softmax(boundary_pred)
            # boundary_pred_ = (boundary_pred_[:,1] > 0.5).int()

            # primitive_embedding, type_per_point = model([coord, normals, offset], edges, boundary_pred_)
            # # loss = criterion(output, target)
            # feat_loss, pull_loss, push_loss = compute_embedding_loss(primitive_embedding, label, offset)
            # type_loss = criterion(type_per_point, semantic)
            # boundary_loss = criterion(boundary_pred, boundary)
            # loss = feat_loss + type_loss + boundary_loss
            onehot_label = label2one_hot(semantic, args['classes'])
            spout, c_idx, c2p_idx_base, type_per_point, primitive_embedding, rec_xyz, rec_label, p_fea, sp_pred_lab, sp_pseudo_lab, sp_pseudo_lab_onehot, normal_loss, sp_center_contrast_loss, p2sp_contrast_loss, param_loss = model([coord, normals, offset], onehot_label, semantic, label, param) # superpoint
            re_xyz_loss = args['w_re_xyz_loss'] * criterion_re_xyz(rec_xyz, coord.transpose(0,1).contiguous())    # compact loss
            if args['re_label_loss'] == 'cel':
                # re_label_loss = args['w_re_label_loss'] * criterion_re_label(rec_label, semantic.unsqueeze(0)) # point label loss
                re_label_loss = args['w_re_label_loss'] * criterion_re_label(rec_label.transpose(0,1).contiguous(), semantic) # point label loss
            elif args['re_label_loss'] == 'mse':
                re_label_loss = args['w_re_label_loss'] * criterion_re_label(rec_label, onehot_label)

            if args['re_sp_loss'] == 'cel':
                re_sp_loss = args['w_re_sp_loss'] * criterion_re_sp(sp_pred_lab, sp_pseudo_lab) # superpoint label loss
            elif args['re_sp_loss'] == 'mse':
                re_sp_loss = args['w_re_sp_loss'] * criterion_re_sp(sp_pred_lab, sp_pseudo_lab_onehot)
            
            # loss = re_xyz_loss + re_label_loss + re_sp_loss + normal_loss
            loss = re_xyz_loss + normal_loss
            # loss = re_xyz_loss


        for bid in range(offset.shape[0]):
            tedg_source = edg_source[bid]
            tedg_target = edg_target[bid]
            tis_transition = is_transition[bid].numpy()
            init_center = c_idx.cpu().numpy()
            filename_ = filename[bid]
            if bid == 0:
                txyz = coord[:offset[0]].cpu().numpy()
                spout_ = spout[:offset[0]].detach().cpu().numpy()
                pt_center_index = c2p_idx_base[:offset[0]].cpu().numpy()
                p_fea_ = p_fea[:offset[0]]
                primitive_embedding_ = primitive_embedding[:offset[0]]
            else:
                txyz = coord[offset[bid-1]:offset[bid]].cpu().numpy()
                spout_ = spout[offset[bid-1]:offset[bid]].detach().cpu().numpy()
                pt_center_index = c2p_idx_base[offset[bid-1]:offset[bid]].cpu().numpy()
                p_fea_ = p_fea[offset[bid-1]:offset[bid]]
                primitive_embedding_ = primitive_embedding[offset[bid-1]:offset[bid]]
            pred_components, pred_in_component, center = get_components(init_center, pt_center_index, spout_, getmax=True)
            # 可视化
            if args.visual:
                # 超点分割结果
                root_name = args.visual_root + '/{}.ply'.format(filename_)
                partition2ply(root_name, txyz, pred_components)
                # 超点均值聚类,基元分割结果
                pred_sp = torch.from_numpy(pred_in_component).to(primitive_embedding_.device)
                p2sp_matrix = label2one_hot(pred_sp, pred_sp.max()+1).squeeze()
                sp_cnt = torch.sum(p2sp_matrix, dim=1)
                sp_fea = torch.matmul(p2sp_matrix, primitive_embedding_) / sp_cnt.unsqueeze(1)
                cluster_pred = sp_mean_shift_gpu(sp_fea, bandwidth = args.bandwidth)
                for i in range(cluster_pred.shape[0]):
                    pred_in_component[pred_in_component == i] = cluster_pred[i]
                coms=np.unique(pred_in_component)
                components = []
                for i in range(len(coms)):
                    te=np.where(pred_in_component==coms[i])
                    components.append(te)
                prim_root_name = args.visual_root + '/{}_prim.ply'.format(filename_)
                partition2ply(prim_root_name, txyz, components)

            # 计算指标
            pred_transition = pred_in_component[tedg_source] != pred_in_component[tedg_target]
            
            # full_pred = perfect_prediction(pred_components, pred_in_component, label_)

            if np.sum(tis_transition) > 0:
                BR_meter.add((tis_transition.sum()) * compute_boundary_recall(tis_transition, 
                            relax_edge_binary(pred_transition, tedg_source, 
                            tedg_target, txyz.shape[0], args['BR_tolerance'])),
                            n=tis_transition.sum())
                BP_meter.add((pred_transition.sum()) * compute_boundary_precision(
                            relax_edge_binary(tis_transition, tedg_source, 
                            tedg_target, txyz.shape[0], args['BR_tolerance']), 
                            pred_transition),n=pred_transition.sum())

        # output = output.max(1)[1]
        n = coord.size(0)
        # if args.multiprocessing_distributed:
        #     loss *= n
        #     count = target.new_tensor([n], dtype=torch.long)
        #     dist.all_reduce(loss), dist.all_reduce(count)
        #     n = count.item()
        #     loss /= n

        # intersection, union, target = intersectionAndUnionGPU(output, semantic, args.classes, args.ignore_label)
        # if args.multiprocessing_distributed:
        #     dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
        # intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        # intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        # accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_re_xyz_meter.update(re_xyz_loss.item(), n)
        loss_re_label_meter.update(re_label_loss.item(), n)
        loss_re_sp_meter.update(re_sp_loss.item(), n)
        loss_norm_meter.update(normal_loss.item(), n)
        loss_meter.update(loss.item(), n)
            
        # spec_cluster_pred = mean_shift_gpu(primitive_embedding, offset, bandwidth=args.bandwidth)
        # s_iou, p_iou = compute_iou(label, spec_cluster_pred, type_per_point, semantic, offset)
        # # All Reduce loss
        # if args.multiprocessing_distributed:
        #     dist.all_reduce(feat_loss.div_(torch.cuda.device_count()))
        #     dist.all_reduce(type_loss.div_(torch.cuda.device_count()))
        #     dist.all_reduce(boundary_loss.div_(torch.cuda.device_count()))
        #     # dist.all_reduce(s_iou.div_(torch.cuda.device_count()))
        #     # dist.all_reduce(p_iou.div_(torch.cuda.device_count()))
        # feat_loss_, type_loss_, boundary_loss_ = feat_loss.data.cpu().numpy(), type_loss.data.cpu().numpy(), boundary_loss.data.cpu().numpy()
        # feat_loss_meter.update(feat_loss_.item())
        # type_loss_meter.update(type_loss_.item())
        # boundary_loss_meter.update(boundary_loss_.item())
        # s_iou_meter.update(s_iou)
        # type_iou_meter.update(p_iou)
        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'LS_re_xyz {loss_re_xyz_meter.val:.4f} '
                        'LS_re_label {loss_re_label_meter.val:.4f} '
                        'LS_re_sp {loss_re_sp_meter.val:.4f} '
                        'LS_norm {loss_norm_meter.val:.4f} '
                        'Loss {loss_meter.val:.4f} '.format(i + 1, len(val_loader),
                                                          data_time=data_time,
                                                          batch_time=batch_time,
                                                          loss_re_xyz_meter=loss_re_xyz_meter,
                                                          loss_re_label_meter=loss_re_label_meter,
                                                          loss_re_sp_meter=loss_re_sp_meter,
                                                          loss_norm_meter=loss_norm_meter,
                                                          loss_meter=loss_meter))

    br = BR_meter.value()[0]
    bp = BP_meter.value()[0]
    f1_score = 2 * br * bp / (br + bp + 1e-10)
    if main_process():
        logger.info('Val result: BR/BP/F1-score {:.4f}/{:.4f}/{:.4f}'.format(
                    br, bp, f1_score))

    # iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    # accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    # mIoU = np.mean(iou_class)
    # mAcc = np.mean(accuracy_class)
    # allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    # if main_process():
    #     logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    #     for i in range(args.classes):
    #         logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
    #     # logger.info('Val result: Seg_mIoU/Type_mIoU {:.4f}/{:.4f}.'.format(s_iou_meter.avg, type_iou_meter.avg))
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')

    return loss_meter.avg, br, bp, f1_score
    # return s_iou_meter.avg, type_iou_meter.avg, feat_loss_meter.avg, type_loss_meter.avg, boundary_loss_meter.avg


if __name__ == '__main__':
    import gc
    gc.collect()
    try:
        main()
    except Exception as e:
        # 添加异常处理，如果出现异常，发送微信通知
        print(str(e))
        headers = {"Authorization": "eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1aWQiOjExNjc4OCwidXVpZCI6ImQzZjBmOGIyLWJmNWMtNDkyMy1hMzMyLTEyN2ViNTg4ZDEyMCIsImlzX2FkbWluIjpmYWxzZSwiaXNfc3VwZXJfYWRtaW4iOmZhbHNlLCJzdWJfbmFtZSI6IiIsInRlbmFudCI6ImF1dG9kbCIsInVwayI6IiJ9.ma-DgwI8MuctNwGWyoVoIVfr7r0Gt64nwJA_U4FToy2lg4ueMhWPlybP0UxP-yKw5_KpfNil4EcWe7t5Wc5Irw"}
        resp = requests.post("https://www.autodl.com/api/v1/wechat/message/send",
            json={
                "title": "A100: The training has stopped",
                "name": "Your network training has made an error",
                "content": str(e)
            }, headers = headers)
