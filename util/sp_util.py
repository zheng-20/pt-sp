import numpy as np
import random
from plyfile import PlyData, PlyElement
import torch

def partition2ply(filename, xyz, components):
    """write a ply with random colors for each components"""
    random_color = lambda: random.randint(0, 255)
    color = np.zeros(xyz.shape)
    for i_com in range(0, len(components)):
        color[components[i_com], :] = [random_color(), random_color()
        , random_color()]
    prop = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1')
    , ('green', 'u1'), ('blue', 'u1')]
    vertex_all = np.empty(len(xyz), dtype=prop)
    for i in range(0, 3):
        vertex_all[prop[i][0]] = xyz[:, i]
    for i in range(0, 3):
        vertex_all[prop[i+3][0]] = color[:, i]
    ply = PlyData([PlyElement.describe(vertex_all, 'vertex')], text=True)
    ply.write(filename)

def get_components(init_center,pt_center_index,asso, getmax=False, trick=False, logger=None):
    #print(asso[0])
    # init_center=init_center.cpu().numpy()
    # pt_center_index=pt_center_index.cpu().numpy()
    # asso=asso.argmax(axis=1).cpu().numpy().reshape(-1,1)
    # init_center=init_center.cpu().numpy()
    # pt_center_index=pt_center_index.cpu().numpy()
    #asso=asso.argmax(axis=1).reshape(-1,1)
    if getmax:
        if trick == False:
            asso=asso.argmax(axis=1).reshape(-1,1)  # n x 1
        else:
            # print(asso.shape)
            mea = np.mean(asso, axis=1)     # n
            # print(mea.shape)
            tmp = np.zeros((mea.shape[0], 1), dtype=np.int32)
            for i in range(asso.shape[0]):
                for j in range(asso.shape[1]):
                    if asso[i, j]*3.0 > mea[i]:
                        tmp[i, 0] = j
                        break
            asso = tmp

    else:
        asso=asso.argmin(axis=1).reshape(-1,1)
    components=[]
    center_com = []

    
    in_component=np.take_along_axis(pt_center_index,asso,1).reshape(-1)
    #sp_index=init_center[in_components].reshape(-1)
    in_component=init_center[in_component]
    coms=np.unique(in_component)
    #   #print(len(init_center))
    # print(len(coms))
    # idx=np.isin(init_center, coms, invert=False)
    # coms=init_center[idx]
    #print(init_center)
    real_center=[]
    for i in range(len(coms)):
        #te=[]
        te=np.where(in_component==coms[i])
        center = np.array(coms[i])
        in_component[te] = i
        components.append(te)
        center_com.append(center)   # center of each component
        
    # real_center=np.array(real_center,dtype=np.int64)
    # real_center=torch.from_numpy(real_center).cuda()
    # print(len(components))
    #logger.info('len components: {}'.format(len(components)))
    #logger.info('len in_component: {}'.format(len(in_component)))
    return components,in_component,center_com

def perfect_prediction(components,in_component,labels):
    """assign each superpoint with the majority label"""
    #print(pred.shape)
    #print(labels.shape)
    full_pred = np.zeros((labels.shape[0],),dtype='uint32')

    for i_com in range(len(components)):
        #print(len(components[i_com]))
        #te=labels[components[i_com]]
        #print(te.shape)
        #label_com = np.argmax(np.bincount(labels[components[i_com]]))
        label_com = labels[components[i_com],1:].sum(0).argmax()
        full_pred[components[i_com]]=label_com


    return full_pred

def relax_edge_binary(edg_binary, edg_source, edg_target, n_ver, tolerance):
    if torch.is_tensor(edg_binary):
        relaxed_binary = edg_binary.cpu().numpy().copy()
    else:
        relaxed_binary = edg_binary.copy()
    transition_vertex = np.full((n_ver,), 0, dtype = 'uint8')
    for i_tolerance in range(tolerance):
        transition_vertex[edg_source[relaxed_binary.nonzero()]] = True
        transition_vertex[edg_target[relaxed_binary.nonzero()]] = True
        relaxed_binary[transition_vertex[edg_source]] = True
        relaxed_binary[transition_vertex[edg_target]>0] = True
    return relaxed_binary