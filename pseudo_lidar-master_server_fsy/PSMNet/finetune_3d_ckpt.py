from __future__ import print_function

import argparse
from calendar import EPOCH
from concurrent.futures.process import _ResultItem
from logging import root
from netrc import netrc
import os
from threading import currentThread
import time
import torch.optim as optim #新增

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

import logger
from dataloader import KITTILoader3D as ls
from dataloader import KITTILoader_dataset3d as DA
from models import *

root_path=os.path.abspath(os.path.join(os.getcwd(), "../../..")) #上三级目录

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--datatype', default='2015',
                    help='datapath') #实际用的2017
""" parser.add_argument('--datapath', default='/media/jiaren/ImageNet/data_scene_flow_2015/training/',
                    help='datapath') """
parser.add_argument('--datapath', default=root_path+'/Dataset/KITTI/object/training/',
                    help='datapath') #新增
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train')
parser.add_argument('--loadmodel', default=root_path+'/Dataset/KITTI/trained/pretrained_sceneflow.tar',
                    help='load model')
parser.add_argument('--savemodel', default=root_path+'/Dataset/psmnet/kitti_3d/', 
                    help='save model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training') #no-cuda和no_cuda 一样
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--lr_scale', type=int, default=50, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--split_file', default=root_path+'/Dataset/KITTI/object/train.txt',
                    help='save model')
parser.add_argument('--btrain', type=int, default=2)
parser.add_argument('--start_epoch', type=int, default=1)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
path_checkpoint = args.savemodel+r'tf-logs/'+'ckpt_best_1.pth'
""" PATH_SAVE=args.savemodel
os.mkdir(args.savemodel + r'tf-logs/') """
# boola = args.no-cuda
#为CPU设置种子用于生成随机数，以使得结果是确定的
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if not os.path.isdir(args.savemodel):
    os.makedirs(args.savemodel)
print(os.path.join(args.savemodel, 'training.log'))
log = logger.setup_logger(os.path.join(args.savemodel, 'training.log'))

all_left_img, all_right_img, all_left_disp, = ls.dataloader(args.datapath,
                                                            args.split_file)

TrainImgLoader = torch.utils.data.DataLoader(
    DA.myImageFloder(all_left_img, all_right_img, all_left_disp, True),
    batch_size=args.btrain, shuffle=True, num_workers=0, drop_last=False)

if args.model == 'stackhourglass':
    model = stackhourglass(args.maxdisp)
elif args.model == 'basic':
    model = basic(args.maxdisp)
else:
    print('no model')

if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()

if args.loadmodel is not None:
    log.info('load model ' + args.loadmodel)
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

optimizer = optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.999))


def train(imgL, imgR, disp_L):
    model.train()
    imgL = Variable(torch.FloatTensor(imgL))
    imgR = Variable(torch.FloatTensor(imgR))
    disp_L = Variable(torch.FloatTensor(disp_L))

    if args.cuda:
        imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()

    # ---------
    mask = (disp_true > 0)
    mask.detach_()
    # ----

    optimizer.zero_grad()

    if args.model == 'stackhourglass':
        output1, output2, output3 = model(imgL, imgR)
        output1 = torch.squeeze(output1, 1)
        output2 = torch.squeeze(output2, 1)
        output3 = torch.squeeze(output3, 1)
        loss = 0.5 * F.smooth_l1_loss(output1[mask], disp_true[mask], size_average=True) + 0.7 * F.smooth_l1_loss(
            output2[mask], disp_true[mask], size_average=True) + F.smooth_l1_loss(output3[mask], disp_true[mask],
                                                                                  size_average=True)
    elif args.model == 'basic':
        output = model(imgL, imgR)
        output = torch.squeeze(output, 1)
        loss = F.smooth_l1_loss(output[mask], disp_true[mask], size_average=True)

    loss.backward()
    optimizer.step()

    # return loss.data[0]
    return loss.item() #修改


def test(imgL, imgR, disp_true):
    model.eval()
    imgL = Variable(torch.FloatTensor(imgL))
    imgR = Variable(torch.FloatTensor(imgR))
    if args.cuda:
        imgL, imgR = imgL.cuda(), imgR.cuda()

    with torch.no_grad():
        output3 = model(imgL, imgR)

    pred_disp = output3.data.cpu()

    # computing 3-px error#
    true_disp = disp_true
    index = np.argwhere(true_disp > 0)
    disp_true[index[0][:], index[1][:], index[2][:]] = np.abs(
        true_disp[index[0][:], index[1][:], index[2][:]] - pred_disp[index[0][:], index[1][:], index[2][:]])
    correct = (disp_true[index[0][:], index[1][:], index[2][:]] < 3) | (
            disp_true[index[0][:], index[1][:], index[2][:]] < true_disp[
        index[0][:], index[1][:], index[2][:]] * 0.05)
    torch.cuda.empty_cache()

    return 1 - (float(torch.sum(correct)) / float(len(index[0])))


def adjust_learning_rate(optimizer, epoch):
    if epoch <= args.lr_scale:
        lr = 0.001
    else:
        lr = 0.0001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    max_acc = 0
    max_epo = 0
    start_full_time = time.time()
    #optimizer = torch.optim.SGD(model.parameters(),lr=0.1)
    # lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[10,20,30,40,50],gamma=0.1)
    #lr_schedule = lr_schedule.load_state_dict(checkpoint['lr_schedule'])#加载lr_scheduler
    start_epoch = 1
    RESUME = False
    
    # 如果接续训练，则加载checkpoint,并初始化训练状态
    if RESUME:
        path_checkpoint = args.savemodel+r'tf-logs/'+'ckpt_best_%s.pth'%(str(epoch)) # 断点路径
        checkpoint = torch.load(path_checkpoint)  # 加载断点
        model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数
        optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
        #lr_schedule.load_state_dict(checkpoint['lr_schedule'])#加载lr_scheduler
        start_epoch = checkpoint['epoch']  # 设置开始的epoch
        
    """ if RESUME:
        path_checkpoint = checkpoint_path #断点路径
        path_checkpoint = args.savemodel+r'tf-logs/'+'/ckpt_best_%s.pth'%(str(epoch)
        checkpoint = torch.load(path_checkpoint) #加载断点
        start_epoch = checkpoint['epoch'] #设置开始的epoch
        model.load_state_dict(checkpoint['net']) #加载模型可学习参数
        optimizer.load_state_dict(checkpoint['optimizer']) #加载优化器参数
        # scheduler.load_state_dict(checkpoint['lr_schedule']) """
        

    for epoch in range(start_epoch, args.epochs + 1):
        total_train_loss = 0
        adjust_learning_rate(optimizer, epoch)
        
       
        ## training ##
        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(TrainImgLoader):
            start_time = time.time()
            loss = train(imgL_crop, imgR_crop, disp_crop_L)
            print('Iter %d training loss = %.3f , time = %.2f' % (batch_idx, loss, time.time() - start_time))
            total_train_loss += loss
        print('epoch %d total training loss = %.3f' % (epoch, total_train_loss / len(TrainImgLoader)))
        
        #设置checkpoint
        min_loss_val=1
        #每隔10个epoch存储
        if epoch %10 == 0:
        #保存到目前为止模型性能最佳的checkpoint；可通过判断语句实现
        # if epoch > int(args.epochs/2) and total_train_loss <= min_loss_val:
            min_loss_val = total_train_loss
            print('epoch:',epoch)
            print('learning rate:',optimizer.state_dict()['param_groups'][0]['lr'])
            checkpoint = {
                'epoch':epoch,
                'net':model.state_dict(),
                #'model_state_dict':net.state_dict(),
                'optimizer':optimizer.state_dict()
                #'lr_schedule': lr_schedule.state_dict()
                }
            if not os.path.isdir(args.savemodel+r'tf-logs/'):
                os.mkdir(args.savemodel+r'tf-logs/')
            torch.save(checkpoint,args.savemodel+r'tf-logs/'+'/ckpt_best_%s.pth'%(str(epoch)))
            print('保存各参数完成，用于后续继续训练。')
        
        # SAVE
        if not os.path.isdir(args.savemodel):
            os.makedirs(args.savemodel)
        savefilename = args.savemodel + '/finetune_' + str(epoch) + '.tar'
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'train_loss': total_train_loss / len(TrainImgLoader),
        }, savefilename)    
    
    
    """ for epoch in range(args.start_epoch, args.epochs + 1):
        total_train_loss = 0
        adjust_learning_rate(optimizer, epoch)
        
       
        ## training ##
        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(TrainImgLoader):
            start_time = time.time()
            loss = train(imgL_crop, imgR_crop, disp_crop_L)
            print('Iter %d training loss = %.3f , time = %.2f' % (batch_idx, loss, time.time() - start_time))
            total_train_loss += loss
        print('epoch %d total training loss = %.3f' % (epoch, total_train_loss / len(TrainImgLoader)))
        
        #设置checkpoint
        min_loss_val=1
        #每隔10个epoch存储
        if epoch %10 == 0:
        #保存到目前为止模型性能最佳的checkpoint；可通过判断语句实现
        # if epoch > int(args.epochs/2) and total_train_loss <= min_loss_val:
            min_loss_val = total_train_loss
            print('epoch:',epoch)
            print('learning rate:',optimizer.state_dict()['param_groups'][0]['lr'])
            checkpoint = {
                'epoch':epoch,
                'net':model.state_dict(),
                #'model_state_dict':net.state_dict(),
                'optimizer':optimizer.state_dict()}
            if not os.path.isdir(args.savemodel+r'tf-logs/'):
                os.mkdir(args.savemodel+r'tf-logs/')
            torch.save(checkpoint,args.savemodel+r'tf-logs/'+'ckpt_best_%s.pth'%(str(epoch)))
            print('保存各参数完成，用于后续继续训练。')
        
        # SAVE
        if not os.path.isdir(args.savemodel):
            os.makedirs(args.savemodel)
        savefilename = args.savemodel + '/finetune_' + str(epoch) + '.tar'
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'train_loss': total_train_loss / len(TrainImgLoader),
        }, savefilename) """

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
