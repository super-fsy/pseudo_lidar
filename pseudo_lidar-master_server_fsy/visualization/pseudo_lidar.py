from pyntcloud import PyntCloud
import pandas as pd
import os
import numpy as np
import PIL.Image as Image
# %matplotlib inline

def load_velo_scan(velo_filename):
    scan = np.fromfile(velo_filename, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    return scan

def paint_points(points, color=[192,0,0]):
    # color = [r, g, b]
    color = np.array([color])
    new_pts = np.zeros([points.shape[0],6])
    new_pts[:,:3] = points
    new_pts[:, 3:] = new_pts[:, 3:] + color
    return new_pts

# from importlib.resources import path
# root_path = os.path.abspath(os.path.join(os.getcwd(), "../../..")) #上三级目录
# root_path = root_path+'/Dataset/Kitti/object/training/pseudo-lidar_velodyne_self_finetune300/' 
# root_path = root_path+'/Dataset/KITTI/object/training/predicted_velodyne_922/' 
# root_path = root_path+'/Dataset/Kitti/object/training/velodyne/' 
# path = root_path+'000001.bin'
path = './000003.bin'

points = load_velo_scan(path)[:,:3]
pd_points = pd.DataFrame(paint_points(points), columns=['x','y','z','red','green','blue'])

cloud = PyntCloud(pd_points)
cloud.plot(initial_point_size=0.02)


