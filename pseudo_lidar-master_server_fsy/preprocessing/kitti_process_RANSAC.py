import argparse
import os

import numpy as np
from sklearn.linear_model import RANSACRegressor

import kitti_util as utils

# 个人电脑配置路径
# root_path=os.path.abspath(os.path.join(os.getcwd(), "../../..")) #上三级目录
# 服务器配置路径
root_path = os.path.abspath(os.path.join(os.getcwd(), "../..")) #上两级目录

def extract_ransac(calib_dir, lidar_dir, planes_dir):
    data_idx_list = [x[:-4] for x in os.listdir(lidar_dir) if x[-4:] == '.bin']

    if not os.path.isdir(planes_dir):
        os.makedirs(planes_dir)

    for data_idx in data_idx_list:

        print('------------- ', data_idx)
        calib = calib_dir + '/' + data_idx + '.txt'
        calib = utils.Calibration(calib)
        pc_velo = lidar_dir + '/' + data_idx + '.bin'
        pc_velo = np.fromfile(pc_velo, dtype=np.float32).reshape(-1, 4)
        pc_rect = calib.project_velo_to_rect(pc_velo[:, :3])
        valid_loc = (pc_rect[:, 1] > 1.5) & \
                    (pc_rect[:, 1] < 1.86) & \
                    (pc_rect[:, 2] > 0) & \
                    (pc_rect[:, 2] < 40) & \
                    (pc_rect[:, 0] > -15) & \
                    (pc_rect[:, 0] < 15)
        pc_rect = pc_rect[valid_loc]
        if len(pc_rect) < 1:
            w = [0, -1, 0]
            h = 1.65
        else:
            reg = RANSACRegressor().fit(pc_rect[:, [0, 2]], pc_rect[:, 1])
            w = np.zeros(3)
            w[0] = reg.estimator_.coef_[0]
            w[2] = reg.estimator_.coef_[1]
            w[1] = -1.0
            h = reg.estimator_.intercept_
            w = w / np.linalg.norm(w)
        print(w)
        print(h)

        lines = ['# Plane', 'Width 4', 'Height 1']

        plane_file = os.path.join(planes_dir, data_idx + '.txt')
        result_lines = lines[:3]
        result_lines.append("{:e} {:e} {:e} {:e}".format(w[0], w[1], w[2], h))
        result_str = '\n'.join(result_lines)
        with open(plane_file, 'w') as f:
            f.write(result_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--calib_dir', default=root_path+'/Dataset/KITTI/object/testing/calib/')
    parser.add_argument('--lidar_dir', default=root_path+'/Dataset/KITTI/object/testing/pseudo-lidar_velodyne_finetune300_self_train/')
    parser.add_argument('--planes_dir', default=root_path+'/Dataset/KITTI/object/testing/pseudo-lidar_planes_finetune300_self_train/')
    args = parser.parse_args()

    extract_ransac(args.calib_dir, args.lidar_dir, args.planes_dir)
