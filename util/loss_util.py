import torch
import torch.nn.functional as F
import numpy as np
from torch import sqrt, exp
import time
import math
from torch.autograd import Variable
from lapsolver import solve_dense
from sklearn.cluster import MeanShift
from lib.pointops.functions import pointops
from sklearn.metrics.pairwise import euclidean_distances

def binarys(points, dep):
    coord_min, coord_max = torch.amin(points, axis=0)[:3], torch.amax(points, axis=0)[:3]
    if dep == 0:
        ax = 1
    else:
        ax = 0
    left, right = coord_min[ax], coord_max[ax]
    half_n = int(points.shape[0] / 2.0)
    pidx1, pidx2 = None, None
    while (left + 1e-8 < right):
        mid = (left + right) / 2.0 
        tmp1 = torch.where((points[:, ax] <= mid))[0]
        if tmp1.size()[0] <= int(half_n*1.1) and tmp1.size()[0] >= int(half_n*0.9):
            tmp2 = torch.where((points[:, ax] > mid))[0]
            pidx1 = tmp1
            pidx2 = tmp2
            break
        elif tmp1.size()[0] < int(half_n*0.9):
            left = mid
        else:
            right = mid
    assert (points.shape[0] == pidx1.size()[0] + pidx2.size()[0])
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

def split_data(points, th=7000):
    # coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
    
    points_idx = torch.arange(points.shape[0])
    pidx_list = dfs(points, points_idx, th, 0)
    pts, indexs = [], []
    for val in pidx_list:
        pts.append(points[val, :])
        indexs.append(val)
    return pts, indexs

def npy(var):
    return var.data.cpu().numpy()

def compute_embedding_loss(pred_feat, gt_label, offset, t_pull=0.5, t_push=1.5):
    '''
    pred_feat: (N, K)
    gt_label: (N)
    '''
    num_pts, feat_dim = pred_feat.shape
    device = pred_feat.device
    pull_loss = torch.Tensor([0.0]).to(device)
    push_loss = torch.Tensor([0.0]).to(device)
    for i in range(len(offset)):
        if i == 0:
            pred = pred_feat[0:offset[i]]
            gt = gt_label[0:offset[i]]
        else:
            pred = pred_feat[offset[i-1]:offset[i]]
            gt = gt_label[offset[i-1]:offset[i]]
        
    # for i in range(batch_size):
        gt = gt - 1
        num_class = gt.max() + 1

        embeddings = []

        for j in range(num_class):
            mask = (gt == j)
            feature = pred[mask]
            if len(feature) == 0:
                continue
            embeddings.append(feature)  # (M, K)

        centers = []

        for feature in embeddings:
            center = torch.mean(feature, dim=0).view(1, -1)
            centers.append(center)

        # intra-embedding loss
        pull_loss_tp = torch.Tensor([0.0]).to(device)
        for feature, center in zip(embeddings, centers):
            dis = torch.norm(feature - center, 2, dim=1) - t_pull
            dis = F.relu(dis)
            pull_loss_tp += torch.mean(dis)

        pull_loss = pull_loss + pull_loss_tp / len(embeddings)

        # inter-embedding loss
        try:
            centers = torch.cat(centers, dim=0)  # (num_class, K)
        except:
            import ipdb
            ipdb.set_trace()

        if centers.shape[0] == 1:
            continue

        dst = torch.norm(centers[:, None, :] - centers[None, :, :], 2, dim=2)

        eye = torch.eye(centers.shape[0]).to(device)
        pair_distance = torch.masked_select(dst, eye == 0)

        pair_distance = t_push - pair_distance
        pair_distance = F.relu(pair_distance)
        push_loss += torch.mean(pair_distance)

    pull_loss = pull_loss / len(offset)
    push_loss = push_loss / len(offset)
    loss = pull_loss + push_loss
    return loss, pull_loss, push_loss

def mean_shift(x, bandwidth):
    # x: [N, f]
    N, c = x.shape
    # IDX = torch.zeros(b, N).to(x.device).long()
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=False, n_jobs=4)
    x_np = x.data.cpu().numpy()
    ms.fit(x_np)
    IDX = ms.labels_
    # for i in range(b):
    #     #print ('Mean shift clustering, might take some time ...')
    #     #tic = time.time()
    #     ms.fit(x_np[i])
    #     #print ('time for clustering', time.time() - tic)
    #     IDX[i] = v(ms.labels_)
    #     cluster_centers = ms.cluster_centers_

    #     num_clusters = cluster_centers.shape[0]
    return IDX

class MeanShift_GPU():
    ''' Do meanshift clustering with GPU support'''
    def __init__(self,bandwidth = 2.5, batch_size = 1000, max_iter = 10, eps = 1e-5, check_converge = False):
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.bandwidth = bandwidth
        self.eps = eps # use for check converge
        self.cluster_eps = 1e-1 # use for check cluster
        self.check_converge = check_converge # Check converge will take 1.5 time longer
          
    def distance_batch(self,a,B):
        ''' Return distance between each element in a to each element in B'''
        return sqrt(((a[None,:] - B[:,None])**2)).sum(2)
    
    def distance(self,a,b):
        return np.sqrt(((a-b)**2).sum())
    
    def fit(self,data):
        with torch.no_grad():
            n = len(data)
            if not data.is_cuda:
                data_gpu = data.cuda()
                X = data_gpu.clone()
            else:
                X = data.clone()
            #X = torch.from_numpy(np.copy(data)).cuda()
            
            for _ in range(self.max_iter):
                max_dis = 0;
                for i in range(0,n,self.batch_size):
                    s = slice(i,min(n,i+ self.batch_size))
                    if self.check_converge:
                        dis = self.distance_batch(X,X[s])
                        max_batch = torch.max(dis)
                        if max_dis < max_batch:
                            max_dis = max_batch;
                        weight = dis
                        weight = self.gaussian(dis, self.bandwidth)
                    else:
                        weight = self.gaussian(self.distance_batch(X,X[s]), self.bandwidth)
                    num = (weight[:,:,None]*X).sum(dim=1)
                    X[s] = num / weight.sum(1)[:,None]                    
                    
                #import pdb; pdb.set_trace()
                #Check converge
                if self.check_converge:
                    if max_dis < self.eps:
                        print("Converged")
                        break
            
            # end_time = time.time()
            # print("algorithm time (s)", end_time- begin_time)
            # Get center and labels
            if True:
                # Convert to numpy cpu show better performance
                points = X.cpu().data.numpy()
                labels, centers = self.cluster_points(points)
            else:
                # use GPU
                labels, centers = self.cluster_points(points)
                
            labels = np.array(labels)
            centers = np.array(centers)
            return labels,centers
        
    def gaussian(self,dist,bandwidth):
        return exp(-0.5*((dist/bandwidth))**2)/(bandwidth*math.sqrt(2*math.pi))
        
    def cluster_points(self, points):
        cluster_ids = []
        cluster_idx = 0
        cluster_centers = []

        for i, point in enumerate(points):
            if(len(cluster_ids) == 0):
                cluster_ids.append(cluster_idx)
                cluster_centers.append(point)
                cluster_idx += 1
            else:
                for j,center in enumerate(cluster_centers):
                    dist = self.distance(point, center)
                    if(dist < self.cluster_eps):
                        cluster_ids.append(j)
                        break
                if(len(cluster_ids) < i + 1):
                    cluster_ids.append(cluster_idx)
                    cluster_centers.append(point)
                    cluster_idx += 1
        return cluster_ids, cluster_centers

def v(var, cuda=True, volatile=False):
    if type(var) == torch.Tensor or type(var) == torch.DoubleTensor:
        res = Variable(var.float(), volatile=volatile)
    elif type(var) == np.ndarray:
        res = Variable(torch.from_numpy(var), volatile=volatile)
    if cuda:
        res = res.cuda()
    return res

def mean_shift_gpu(x, offset, bandwidth):
    # x: [N, f]
    N, c = x.shape
    # IDX = torch.zeros(N).to(x.device).long()
    IDX = np.zeros(N, dtype=np.int)
    ms = MeanShift_GPU(bandwidth = bandwidth, batch_size = 700, max_iter = 10, eps = 1e-5, check_converge = False)
    for i in range(len(offset)):
        if i == 0:
            pred = x[0:offset[i]]
        else:
            pred = x[offset[i-1]:offset[i]]
    # for i in range(b):
        # print ('Mean shift clustering, might take some time ...')
        tic = time.time()
        labels, centers = ms.fit(pred)
        # print ('[{}/{}] time for Mean shift clustering'.format(i+1, len(offset)), time.time() - tic)
        if i == 0:
            # IDX[0:offset[i]] = v(labels)
            IDX[0:offset[i]] = labels
        else:
            # IDX[offset[i-1]:offset[i]] = v(labels)
            IDX[offset[i-1]:offset[i]] = labels
        # IDX[i] = v(labels)
        # cluster_centers = centers

        # num_clusters = cluster_centers.shape[0]
        # print(num_clusters)
    return IDX

def block_mean_shift_gpu(coord, x, offset, bandwidth):
    # x: [N, f]
    N, c = x.shape
    # IDX = torch.zeros(N).to(x.device).long()
    IDX = np.zeros(N, dtype=np.int)
    ms = MeanShift_GPU(bandwidth=bandwidth, batch_size=1000)
    for i in range(len(offset)):
        if i == 0:
            pred = x[0:offset[i]]
            coord_i = coord[0:offset[i]]
        else:
            pred = x[offset[i-1]:offset[i]]
            coord_i = coord[offset[i-1]:offset[i]]

        # print ('Mean shift clustering, might take some time ...')
        # tic = time.time()
        # global_labels, global_centers = ms.fit(pred)
        # print ('[{}/{}] time for Mean shift clustering'.format(i+1, len(offset)), time.time() - tic)
        tic = time.time()
        xs, indexs = split_data(coord_i, th=7000)   # 点云分块
        labels = []
        centers = []
        for indexs_i in indexs:
            labels_i, centers_i = ms.fit(pred[indexs_i])
            labels.append(labels_i)
            centers.append(centers_i)
        # # 可视化部分点云
        # primitive_colors = np.random.rand(10000, 3)
        # fp = open('visual/0_0.obj', 'w')
        # for j in range(coord_i[indexs[0]].shape[0]):
        #     v = coord_i[indexs[0]][j]
        #     if labels[0][j] < 0:
        #         p = np.array([0,0,0])
        #     else:
        #         p = primitive_colors[labels[0][j]]
        #     fp.write('v %f %f %f %f %f %f\n'%(v[0],v[1],v[2],p[0],p[1],p[2]))
        # fp.close()
        combined_labels = np.zeros(pred.shape[0], dtype=np.int) # 合并聚类结果
        n_blocks = len(xs)
        block_labels = []   # 每个块的原始聚类标签，用于合并聚类查找标签号
        threshold = 0.7   # 聚类合并阈值
        for j in range(n_blocks):
            block_labels.append(np.arange(labels[j].min(), labels[j].max() + 1))
            if j == n_blocks - 1:
                break
            labels[j+1] += labels[j].max() + 1
        for j in range(n_blocks):
            for k in range(j+1, n_blocks):
                distances = euclidean_distances(centers[j], centers[k])
                for l in range(len(distances)):
                    if True in (distances[l] < threshold):
                        same = np.where(distances[l] < threshold)
                        for s in same[0]:
                            labels[k][labels[k] == block_labels[k][s]] = block_labels[j][l]
                            block_labels[k][s] = block_labels[j][l]
        for j in range(len(indexs)):
            combined_labels[indexs[j]] = labels[j]
        # # 可视化合并聚类点云
        # primitive_colors = np.random.rand(10000, 3)
        # fp = open('visual/0.obj', 'w')
        # for j in range(coord_i.shape[0]):
        #     v = coord_i[j]
        #     if combined_labels[j] < 0:
        #         p = np.array([0,0,0])
        #     else:
        #         p = primitive_colors[combined_labels[j]]
        #     fp.write('v %f %f %f %f %f %f\n'%(v[0],v[1],v[2],p[0],p[1],p[2]))
        # fp.close()
        print ('[{}/{}] time for Blocked Mean shift clustering'.format(i+1, len(offset)), time.time() - tic)
        if i == 0:
            # IDX[0:offset[i]] = v(labels)
            IDX[0:offset[i]] = combined_labels
        else:
            # IDX[offset[i-1]:offset[i]] = v(labels)
            IDX[offset[i-1]:offset[i]] = combined_labels
        # IDX[i] = v(labels)
        # cluster_centers = centers

        # num_clusters = cluster_centers.shape[0]
        # print(num_clusters)
    return IDX

def sp_mean_shift_gpu(x, batch_size=50, bandwidth=1.3):
    # x: [N, f]
    N, c = x.shape
    IDX = np.zeros(N, dtype=np.int)
    ms = MeanShift_GPU(bandwidth=bandwidth, batch_size=batch_size)
    labels, centers = ms.fit(x)
    # IDX = labels
    return labels

def mean_IOU_primitive_segment(matching, predicted_labels, labels, pred_prim, gt_prim):
	"""
	Primitive type IOU, this is calculated over the segment level.
	First the predicted segments are matched with ground truth segments,
	then IOU is calculated over these segments.
	:param matching
	:param pred_labels: N x 1, pred label id for segments
	:param gt_labels: N x 1, gt label id for segments
	:param pred_prim: K x 1, pred primitive type for each of the predicted segments
	:param gt_prim: N x 1, gt primitive type for each point
	"""
	batch_size = labels.shape[0]
	IOU = []
	IOU_prim = []

	for b in range(batch_size):
		iou_b = []
		iou_b_prim = []
		iou_b_prims = []
		len_labels = np.unique(predicted_labels[b]).shape[0]
		rows, cols = matching[b]
		count = 0
		for r, c in zip(rows, cols):
			pred_indices = predicted_labels[b] == r
			gt_indices = labels[b] == c

			# use only matched segments for evaluation
			if (np.sum(gt_indices) == 0) or (np.sum(pred_indices) == 0):
				continue

			# also remove the gt labels that are very small in number
			if np.sum(gt_indices) < 100:
				continue

			iou = np.sum(np.logical_and(pred_indices, gt_indices)) / (
						np.sum(np.logical_or(pred_indices, gt_indices)) + 1e-8)
			iou_b.append(iou)

			# evaluation of primitive type prediction performance
			gt_prim_type_k = gt_prim[b][gt_indices][0]
			# if gt_prim[b][gt_indices][0] != np.argmax(np.bincount(gt_prim[b][gt_indices])):
			# 	gt_prim_type_k = np.argmax(np.bincount(gt_prim[b][gt_indices]))
			try:
				predicted_prim_type_k = pred_prim[b][r]
			except:
				import ipdb;
				ipdb.set_trace()

			iou_b_prim.append(gt_prim_type_k == predicted_prim_type_k)
			iou_b_prims.append([gt_prim_type_k, predicted_prim_type_k])

		# find the mean of IOU over this shape
		IOU.append(np.mean(iou_b))
		IOU_prim.append(np.mean(iou_b_prim))
	return np.mean(IOU), np.mean(IOU_prim), iou_b_prims

def to_one_hot(target_t, maxx=500):
	N = target_t.shape[0]
	maxx = np.max(target_t) + 1
	if maxx <= 0:
		maxx = 1
	target_one_hot = np.zeros((N, maxx))

	for i in range(target_t.shape[0]):
		if target_t[i] >= 0:
			target_one_hot[i][target_t[i]] = 1
	#target_one_hot = target_one_hot.scatter_(1, target_t.long(), 1)

	target_one_hot = torch.from_numpy(target_one_hot)
	return target_one_hot

def relaxed_iou_fast(pred, gt, max_clusters=500):
	batch_size, N, K = pred.shape
	normalize = torch.nn.functional.normalize
	one = torch.ones(1)

	norms_p = torch.unsqueeze(torch.sum(pred, 1), 2)
	norms_g = torch.unsqueeze(torch.sum(gt, 1), 1)
	cost = []

	for b in range(batch_size):
		p = pred[b]
		g = gt[b]
		c_batch = []
		dots = p.transpose(1, 0) @ g
		r_iou = dots
		r_iou = r_iou / (norms_p[b] + norms_g[b] - dots + 1e-7)
		cost.append(r_iou)
	cost = torch.stack(cost, 0)
	return cost

def primitive_type_segment_torch(pred, weights):
	"""
	Returns the primitive type for every segment in the predicted shape.
	:param pred: N x L
	:param weights: N x k
	"""
	d = torch.unsqueeze(pred, 2).float() * torch.unsqueeze(weights, 1).float()
	d = torch.sum(d, 0)
	return torch.max(d, 0)[1]

def SIOU_matched_segments(target, pred_labels, primitives_pred, primitives, weights):
	"""
	Computes iou for segmentation performance and primitive type
	prediction performance.
	First it computes the matching using hungarian matching
	between predicted and ground truth labels.
	Then it computes the iou score, starting from matching pairs
	coming out from hungarian matching solver. Note that
	it is assumed that the iou is only computed over matched pairs.
	That is to say, if any column in the matched pair has zero
	number of points, that pair is not considered.
	
	It also computes the iou for primitive type prediction. In this case
	iou is computed only over the matched segments.
	"""
	# 2 is open spline and 9 is close spline
	primitives[primitives == 0] = 9
	primitives[primitives == 6] = 9
	primitives[primitives == 7] = 9
	primitives[primitives == 8] = 2

	primitives_pred[primitives_pred == 0] = 9
	primitives_pred[primitives_pred == 6] = 9
	primitives_pred[primitives_pred == 7] = 9
	primitives_pred[primitives_pred == 8] = 2

	labels_one_hot = to_one_hot(target)
	cluster_ids_one_hot = to_one_hot(pred_labels)

	cost = relaxed_iou_fast(torch.unsqueeze(cluster_ids_one_hot, 0).float(), torch.unsqueeze(labels_one_hot, 0).float())
	cost_ = 1.0 - cost.data.cpu().numpy()
	matching = []

	for b in range(1):
		rids, cids = solve_dense(cost_[b])
		matching.append([rids, cids])

	primitives_pred_hot = to_one_hot(primitives_pred, 10).float()

	# this gives you what primitive type the predicted segment has.
	prim_pred = primitive_type_segment_torch(primitives_pred_hot, weights).data.cpu().numpy()
	target = np.expand_dims(target, 0)
	pred_labels = np.expand_dims(pred_labels, 0)
	prim_pred = np.expand_dims(prim_pred, 0)
	primitives = np.expand_dims(primitives, 0)

	segment_iou, primitive_iou, iou_b_prims = mean_IOU_primitive_segment(matching, pred_labels, target, prim_pred,
																		 primitives)
	return segment_iou, primitive_iou, matching, iou_b_prims

def compute_iou(label, spec_cluster_pred, type_per_point, semantic, offset):
    test_s_iou = []
    test_p_iou= []
    type_per_point = torch.argmax(type_per_point, dim=-1)
    label, type_per_point, semantic = npy(label), npy(type_per_point), npy(semantic)
    for i in range(len(offset)):
        if i == 0:
            seg_label = label[0:offset[i]]
            seg_pred = spec_cluster_pred[0:offset[i]]
            type_pred = type_per_point[0:offset[i]]
            type_label = semantic[0:offset[i]]
        else:
            seg_label = label[offset[i-1]:offset[i]]
            seg_pred = spec_cluster_pred[offset[i-1]:offset[i]]
            type_pred = type_per_point[offset[i-1]:offset[i]]
            type_label = semantic[offset[i-1]:offset[i]]
        
        weights = to_one_hot(seg_pred, np.unique(seg_pred).shape[0])
        s_iou, p_iou, _, _ = SIOU_matched_segments(
		seg_label,
		seg_pred,
		type_pred,
		type_label,
		weights,
        )
        test_s_iou.append(s_iou)
        test_p_iou.append(p_iou)
    
    return np.mean(test_s_iou), np.mean(test_p_iou)

def compute_iou_RG(gt_face_labels, face_labels, semantic_faces, semantic_faces_gt):
    weights = to_one_hot(face_labels, np.unique(face_labels).shape[0])
    s_iou, p_iou, _, _ = SIOU_matched_segments(
		gt_face_labels,
		face_labels,
		semantic_faces,
		semantic_faces_gt,
		weights,
	)
    return s_iou, p_iou

def compute_boundary_loss_for_type(p, features, target, offset):

    total_loss = torch.Tensor([0.0]).to(features.device)
    cnt = 0

    for i in range(len(offset)):
        if i == 0:
            pred = features[0:offset[i]]
            gt = target[0:offset[i]]
            p_i = p[0:offset[i]]
        else:
            pred = features[offset[i-1]:offset[i]]
            gt = target[offset[i-1]:offset[i]]
            p_i = p[offset[i-1]:offset[i]]
        
        valid_class = (gt != -1)  # remove background
        gt = gt[valid_class]
        pred = pred[valid_class]
        p_i = p_i[valid_class]

        labels = F.one_hot(gt, gt.max()+1).float()

        nsample = 8
        # p_i = torch.unsqueeze(p_i, dim=0)
        offset_i = torch.tensor(p_i.shape[0]).to(features.device).int()
        neighbor_idx_i = pointops.knnquery(nsample, p_i, p_i, offset_i, offset_i)
        nsample -= 1

        neighbor_idx_i = neighbor_idx_i[0][..., 1:].contiguous()  # [m, nsample-1]
        m = neighbor_idx_i.shape[0]

        neighbor_label = labels[neighbor_idx_i.view(-1).long(), :].view(m, nsample, labels.shape[1]) # (m, nsample, ncls)
        neighbor_feature = pred[neighbor_idx_i.view(-1).long(), :].view(m, nsample, pred.shape[1])

        labels = torch.argmax(torch.unsqueeze(labels, -2), -1)  # [m, 1]
        neighbor_label = torch.argmax(neighbor_label, -1)  # [m, nsample]
        posmask = labels == neighbor_label  # [m, nsample]

        point_mask = torch.sum(posmask.int(), -1)  # (m)
        point_mask = torch.logical_and(0 < point_mask, point_mask < nsample)    # 边界点mask

        if not torch.any(point_mask):
            loss = .0
            total_loss += loss
            cnt += 1
            continue

        posmask = posmask[point_mask]
        pred = pred[point_mask]    # 边界点的特征
        neighbor_feature = neighbor_feature[point_mask]

        # dist_l2
        dist = torch.unsqueeze(pred, -2) - neighbor_feature
        dist = torch.sqrt(torch.sum(dist ** 2, axis=-1) + 1e-12)

        # # dist_kl
        # pred = F.log_softmax(pred, dim=-1)
        # pred = pred.unsqueeze(-2)
        # pred = pred.expand([neighbor_feature.shape[0], nsample, 128])
        # neighbor_feature = F.log_softmax(neighbor_feature, dim=-1)
        # dist = F.kl_div(neighbor_feature, pred, reduction='none', log_target=True)
        # dist = F.softmax(neighbor_feature, dim=-1) * (F.log_softmax(neighbor_feature, dim=-1) - F.log_softmax(pred, dim=-1))
        # dist = dist.sum(-1) # 边界点与邻域点的kl散度度量

        # compute loss
        dist = -dist
        dist = dist - torch.max(dist, -1, keepdim=True)[0]  # NOTE: max return both (max value, index)

        dist = dist / 1
        exp = torch.exp(dist)
        # if invalid_mask is not None:
        #     valid_mask = 1 - invalid_mask
        #     exp = exp * valid_mask

        # softnn
        pos = torch.sum(exp * posmask, axis=-1)  # (m)
        neg = torch.sum(exp, axis=-1)  # (m)
        loss = -torch.log(pos / neg + 1e-12)

        # # nce
        # neg = torch.sum(exp * (~posmask), axis=-1)  # (m)
        # pos = torch.sum(exp * posmask, axis=-1)  # (m)
        # exp = torch.sum(exp, axis=-1)  # (m)
        # loss = (pos / (exp + neg) + 1e-12)
        # loss = -torch.log(loss)

        loss = torch.mean(loss)

        total_loss += loss
        cnt += 1
    
    total_loss = total_loss / cnt

    return total_loss
