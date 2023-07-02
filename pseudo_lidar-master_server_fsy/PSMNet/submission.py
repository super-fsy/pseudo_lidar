
from __future__ import print_function
import os
#指定GPU运行脚本
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import argparse
# import os
import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import time
import math
from models import *
from PIL import Image
#下面为新增
import skimage
from utils import preprocess 
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

#  个人电脑配置路径
# root_path=os.path.abspath(os.path.join(os.getcwd(), "../../..")) #上三级目录
#  服务器配置路径
root_path = os.path.abspath(os.path.join(os.getcwd(), "../../..")) #上两级目录

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--KITTI', default='2015',
                    help='KITTI version')
parser.add_argument('--datapath', default= root_path+'/Dataset/KITTI/object/training/',
                    help='select model')
parser.add_argument('--loadmodel', default=root_path+'/Dataset/psmnet/pretrained_model_backup/finetune_300_self_train.tar',
                    help='loading model')
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--maxdisp', default=192,
                    help='maxium disparity')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--save_path', type=str, default=root_path+'/Dataset/KITTI/object/training/predict_disparity_finetune300_self_train', metavar='S',
                    help='path to save the predict')
parser.add_argument('--save_figure', action='store_true', help='if true, save the numpy file, not the png file')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.KITTI == '2015':
   from dataloader import KITTI_submission_loader as DA
else:
   from dataloader import KITTI_submission_loader2012 as DA  

test_left_img, test_right_img = DA.dataloader(args.datapath)

if args.model == 'stackhourglass':
    model = stackhourglass(args.maxdisp)
elif args.model == 'basic':
    model = basic(args.maxdisp)
else:
    print('no model')

model = nn.DataParallel(model, device_ids=[0])
model.cuda()

if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

def test(imgL,imgR):
    model.eval()

    if args.cuda:
        imgL = imgL.cuda()
        imgR = imgR.cuda()     

    with torch.no_grad():
        output = model(imgL,imgR)
    output = torch.squeeze(output).data.cpu().numpy()
    return output

#下面来自submission_1
# def test(imgL,imgR):
#         model.eval()

#         if args.cuda:
#            imgL = torch.FloatTensor(imgL).cuda()
#            imgR = torch.FloatTensor(imgR).cuda()     

#         imgL, imgR= Variable(imgL), Variable(imgR)

#         with torch.no_grad():
#             output = model(imgL,imgR)
#         output = torch.squeeze(output)
#         pred_disp = output.data.cpu().numpy()

#         return pred_disp

def main():
    
    # processed = preprocess.get_transform(augment=False) #新增 来自submission_origin

    if not os.path.isdir(args.save_path):
       os.makedirs(args.save_path)
       
    normal_mean_var = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}
    infer_transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(**normal_mean_var)])    

    for inx in range(len(test_left_img)):
        #生成定量视差图检验效果 暂定为10张图片 inx==10
        # if inx==60:
        #     break
        imgL_o = Image.open(test_left_img[inx]).convert('RGB')
        imgR_o = Image.open(test_right_img[inx]).convert('RGB')

        imgL = infer_transform(imgL_o)
        imgR = infer_transform(imgR_o)         

        # pad to width and hight to 16 times
        if imgL.shape[1] % 16 != 0:
            times = imgL.shape[1]//16       
            top_pad = (times+1)*16 -imgL.shape[1]
        else:
            top_pad = 0

        if imgL.shape[2] % 16 != 0:
            times = imgL.shape[2]//16                       
            right_pad = (times+1)*16-imgL.shape[2]
        else:
            right_pad = 0    

        imgL = F.pad(imgL,(0,right_pad, top_pad,0)).unsqueeze(0)
        imgR = F.pad(imgR,(0,right_pad, top_pad,0)).unsqueeze(0)

        start_time = time.time()
        pred_disp = test(imgL,imgR)
        print('time = %.2f' %(time.time() - start_time))

        if top_pad !=0 or right_pad != 0:
            img = pred_disp[top_pad:,:-right_pad]
        else:
            img = pred_disp

        img = (img*256).astype('uint16')
        img = Image.fromarray(img)
        # img.save(test_left_img[inx].split('/')[-1])
        
        if args.save_figure:
           skimage.io.imsave(args.save_path+'/'+test_left_img[inx].split('/')[-1],(img*256).astype('uint16'))
        else:
           np.save(args.save_path+'/'+test_left_img[inx].split('/')[-1][:-4], img)
        
    #来自submission_origin
"""     for inx in range(len(test_left_img)):

       imgL_o = (skimage.io.imread(test_left_img[inx]).astype('float32'))
       imgR_o = (skimage.io.imread(test_right_img[inx]).astype('float32'))
       imgL = processed(imgL_o).numpy()
       imgR = processed(imgR_o).numpy()
       imgL = np.reshape(imgL,[1,3,imgL.shape[1],imgL.shape[2]])
       imgR = np.reshape(imgR,[1,3,imgR.shape[1],imgR.shape[2]])
       
       # pad to (384, 1248)
       top_pad = 384-imgL.shape[2]
       left_pad = 1248-imgL.shape[3]
       imgL = np.lib.pad(imgL,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)
       imgR = np.lib.pad(imgR,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)

       start_time = time.time()
       pred_disp = test(imgL,imgR)
       print('time = %.2f' %(time.time() - start_time))

       top_pad   = 384-imgL_o.shape[0]
       left_pad  = 1248-imgL_o.shape[1]
       img = pred_disp[top_pad:,:-left_pad]
       print(test_left_img[inx].split('/')[-1])
       if args.save_figure:
           skimage.io.imsave(args.save_path+'/'+test_left_img[inx].split('/')[-1],(img*256).astype('uint16'))
       else:
           np.save(args.save_path+'/'+test_left_img[inx].split('/')[-1][:-4], img) """


if __name__ == '__main__':
    main()







