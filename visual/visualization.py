import os,sys
import numpy as np

def write_obj(pc, pred, color, name):
    # time_tag = time.strftime("%Y%m%d-%H%M%S")
    file_name = name
    fp = open('/home/fz20/file/project/point-transformer-boundary/visual/val_vis/{}.obj'.format(file_name), 'w')
    for j in range(pc.shape[0]):
        v = pc[j]
        if pred[j] < 0:
            p = np.array([0,0,0])
        else:
            p = color[pred[j]]
        fp.write('v %f %f %f %f %f %f\n'%(v[0],v[1],v[2],p[0],p[1],p[2]))
    fp.close()
    # fp = open('/home/fz20/project/point-transformer-boundary/visual/{}_{}.obj'.format(type+'_gt', time_tag), 'w')
    # for j in range(pc.shape[0]):
    #     v = pc[j]
    #     if gt[j] < 0:
    #         p = np.array([0,0,0])
    #     else:
    #         p = color[gt[j]]
    #     fp.write('v %f %f %f %f %f %f\n'%(v[0],v[1],v[2],p[0],p[1],p[2]))
    # fp.close()


data_root = '/data/fz20/dataset/ABC/val_final/'
data_list = sorted(os.listdir(data_root))
bound_color = np.array([[0.41176,0.41176,0.41176], [1,0,0]])
for item in data_list:
    data_path = os.path.join(data_root, item)
    data = np.load(data_path)
    coord, normals, boundary, label, semantic, param, F, edges = data['V'],data['N'],data['B'],data['L'],data['S'],data['T_param'],data['F'],data['edges']
    write_obj(coord, boundary, bound_color, name=item[:-4])
    