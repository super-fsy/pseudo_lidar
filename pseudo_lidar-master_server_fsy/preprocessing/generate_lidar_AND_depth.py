import argparse
import os
#指定GPU运行脚本
# os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"

import numpy as np
import scipy.misc as ssc

import kitti_util

# 个人电脑配置路径
# root_path=os.path.abspath(os.path.join(os.getcwd(), "../../..")) #上三级目录
# 服务器配置路径
root_path = os.path.abspath(os.path.join(os.getcwd(), "../../..")) #上三级目录

#from视差图to深度图
def project_disp_to_depth(calib, disp):
    disp[disp < 0] = 0
    baseline = 0.54
    mask = disp > 0
    depth = calib.f_u * baseline / (disp + 1. - mask)
    # rows, cols = depth.shape
    return depth
    
def project_disp_to_points(calib, disp, max_high):
    disp[disp < 0] = 0
    baseline = 0.54
    mask = disp > 0
    depth = calib.f_u * baseline / (disp + 1. - mask)
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    points = np.stack([c, r, depth])
    points = points.reshape((3, -1))
    points = points.T
    points = points[mask.reshape(-1)]
    cloud = calib.project_image_to_velo(points)
    valid = (cloud[:, 0] >= 0) & (cloud[:, 2] < max_high)
    return cloud[valid]

def project_depth_to_points(calib, depth, max_high):
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    points = np.stack([c, r, depth])
    points = points.reshape((3, -1))
    points = points.T
    cloud = calib.project_image_to_velo(points)
    valid = (cloud[:, 0] >= 0) & (cloud[:, 2] < max_high)
    return cloud[valid]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Libar')
    parser.add_argument('--calib_dir', type=str,
                        default=root_path+'/Dataset/KITTI/object/testing/calib/')
    parser.add_argument('--disparity_dir', type=str,
                        default=root_path+'/Dataset/KITTI/object/testing/predict_disparity_finetune300_self_train')
    parser.add_argument('--save_dir', type=str,
                        default=root_path+'/Dataset/KITTI/object/testing/pseudo-lidar_velodyne_finetune300_self_train/')
    parser.add_argument('--max_high', type=int, default=1)
    parser.add_argument('--is_depth', action='store_true')
    # parser.add_argument('--is_depth', action='store_false')

    args = parser.parse_args()

    assert os.path.isdir(args.disparity_dir)
    assert os.path.isdir(args.calib_dir)

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    disps = [x for x in os.listdir(args.disparity_dir) if x[-3:] == 'png' or x[-3:] == 'npy']
    disps = sorted(disps)

    for fn in disps:
        predix = fn[:-4]
        calib_file = '{}/{}.txt'.format(args.calib_dir, predix)
        calib = kitti_util.Calibration(calib_file)
        # disp_map = ssc.imread(args.disparity_dir + '/' + fn) / 256.
        if fn[-3:] == 'png':
            disp_map = ssc.imread(args.disparity_dir + '/' + fn)
        elif fn[-3:] == 'npy':
            disp_map = np.load(args.disparity_dir + '/' + fn)
        else:
            assert False
        if not args.is_depth:
            disp_map = (disp_map).astype(np.uint16)/256.
            # disp_map = (disp_map).astype(np.float32)/256.
            # disp_map = (disp_map).astype(np.uint16)
            # depth_map = project_disp_to_depth(calib,disp_map)
            lidar = project_disp_to_points(calib, disp_map, args.max_high)
        else:
            disp_map = (disp_map).astype(np.float32)/256.
            # disp_map = (disp_map).astype(np.float32)
            # depth_map = project_disp_to_depth(calib,disp_map)
            lidar = project_depth_to_points(calib, disp_map, args.max_high)
        # pad 1 in the indensity dimension 并保存点云
        lidar = np.concatenate([lidar, np.ones((lidar.shape[0], 1))], 1)
        lidar = lidar.astype(np.float32)
        lidar.tofile('{}/{}.bin'.format(args.save_dir, predix))
        print('Finish Lidar {}'.format(predix))
        #将其保存为深度图  用np.save()函数方可以使用load()可视化
        # np.save(args.save_dir + '/' + predix, depth_map)
        # depth_map.tofile('{}/{}.npy'.format(args.save_dir, predix))
        # depth_map.tofile('{}/{}.png'.format(args.save_dir, predix))
        # print('Finish Depth {}'.format(predix))

