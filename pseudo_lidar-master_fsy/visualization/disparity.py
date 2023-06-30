#可实现视差图和深度图的可视化，并可将图像灰度化及保存
from pipes import Template
import numpy as np
import matplotlib.pyplot as plt
import os
root_path=os.path.abspath(os.path.join(os.getcwd(), "../../..")) #
# data_path=root_path+'/Dataset/KITTI/object/training/disparities/'
# data_path=root_path+'/Dataset/KITTI/object/training/disparity_psmnet/'
# data_path=root_path+'/Dataset/KITTI/object/testing/predict_disparity_sceneflow/'
# data_path=root_path+'/Dataset/KITTI/object/training/predict_velodyne_926/'
# data_path=root_path+'/Dataset/KITTI/object/training/depth-map_921/'
data_path=root_path+'/Dataset/KITTI/object/training/predict_disparity/'
TempPath=os.path.abspath(os.path.join(data_path,".."))
#python获取文件路径特定部分的方法https://www.codenong.com/22997264/
file=os.path.relpath(data_path, TempPath)
save_path=root_path+'/Dataset/KITTI/object/predicted_disparity/testing/'
disparity_number=10
disparity=data_path+'0000'+'%s.npy'%(str(disparity_number))
disparitymap=np.load(disparity)
plt.imshow(disparitymap)
#plt.savefig(save_path + 'disparitymap.jpg')
plt.savefig(save_path + file + '_' + str(disparity_number) + '.png')       #执行后可以将文件保存为jpg格式图像，可以双击直接查看。也可以不执行这一行，直接执行下一行命令进行可视化。但是如果要使用命令行将其保存，则需要将这行代码置于下一行代码之前，不然保存的图像是空白的
plt.show()                        #在线显示图像
# disparity_1=data_path_1+'00000'+'%s.npy'%(str(disparity_number))
# disparitymap_1=np.load(disparity_1)
# plt.imshow(disparitymap_1)
# plt.show()                        #在线显示图像  图像对比
# for file in data_path:
#         if not os.path.isdir(file): #判断是否是文件夹，不是文件夹才打开
#           disparitymap = np.load(file); #打开文件
#           plt.imshow(disparitymap)
#           plt.show()  

# path_checkpoint = args.savemodel+r'tf-logs/'+'ckpt_best_%s.pth'%(str(start_epoch))
""" depthmap = np.load('0000.npy')    #使用numpy载入npy文件
plt.imshow(depthmap)              #执行这一行后并不会立即看到图像，这一行更像是将depthmap载入到plt里
# plt.colorbar()                   #添加colorbar
plt.savefig('depthmap.jpg')       #执行后可以将文件保存为jpg格式图像，可以双击直接查看。也可以不执行这一行，直接执行下一行命令进行可视化。但是如果要使用命令行将其保存，则需要将这行代码置于下一行代码之前，不然保存的图像是空白的
plt.show()                        #在线显示图像
"""
#若要将图像存为灰度图，可以执行如下两行代码
# import scipy.misc
# import imageio #新增 
# scipy.misc.imsave("depth.png", depthmap)
# plt.savefig(data_path+disparity_number+"depth.png") 
# imageio.imsave(data_path+"depth%s.png"%(str(disparity_number)),disparitymap ) 