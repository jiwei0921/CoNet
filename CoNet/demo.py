# coding:utf-8
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dataset_loader import MyData, MyTestData
from model import ResNet101,Integration
from collector import TriAtt
from functions import imsave
from trainer import Trainer

import os
import argparse
import time

configurations = {
    1: dict(
        max_iteration=2000000,
        lr=1.0e-10,
        momentum=0.99,
        weight_decay=0.0005,
        spshot=20000,
        nclass=2,
        sshow=10,
    )}

parameters = {
    "phase":"test",                    # train or test
    "param":True,                      # True or False
    "dataset":"NJUD",                 # DUT-RGBD, NJUD, NLPR, STEREO, LFSD, RGBD135
    "snap_num":str(1200000)+'.pth',    # Snapshot Number
}


parser=argparse.ArgumentParser()
parser.add_argument('--phase', type=str, default=parameters["phase"], help='train or test')
parser.add_argument('--param', type=str, default=parameters["param"], help='path to pre-trained parameters')
parser.add_argument('--train_dataroot', type=str, default='/dockerdata/weiji/Code/Data/E_Depth/train_data/', help='path to train data')
parser.add_argument('--test_dataroot', type=str, default='/dockerdata/weiji/Code/Data/E_Depth/test_data/'+ parameters["dataset"], help='path to test data')
parser.add_argument('--snapshot_root', type=str, default='../Out/snapshot', help='path to snapshot')
parser.add_argument('--salmap_root', type=str, default='../Out/sal_map', help='path to saliency map')
parser.add_argument('--out1', type=str, default='../Out/edge', help='path to saliency map')
parser.add_argument('--out2', type=str, default='../Out/depth', help='path to saliency map')
parser.add_argument('--out3', type=str, default='../Out/sal_att', help='path to saliency map')
parser.add_argument('-c', '--config', type=int, default=1, choices=configurations.keys())
args = parser.parse_args()
cfg = configurations[args.config]
os.environ["CUDA_VISIBLE_DEVICES"]="0"
cuda = torch.cuda.is_available()


"""""""""""~~~ dataset loader ~~~"""""""""
train_dataRoot = args.train_dataroot
test_dataRoot = args.test_dataroot
if not os.path.exists(args.snapshot_root):
    os.mkdir(args.snapshot_root)
if not os.path.exists(args.salmap_root):
    os.mkdir(args.salmap_root)
if not os.path.exists(args.out1):
    os.mkdir(args.out1)
if not os.path.exists(args.out2):
    os.mkdir(args.out2)
if not os.path.exists(args.out3):
    os.mkdir(args.out3)

if args.phase == 'train':
    SnapRoot = args.snapshot_root       # checkpoint
    train_loader = torch.utils.data.DataLoader(MyData(train_dataRoot, transform=True),batch_size=2, shuffle=True, num_workers=4, pin_memory=True)
else:
    MapRoot = args.salmap_root
    test_loader = torch.utils.data.DataLoader(MyTestData(test_dataRoot, transform=True),batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

print ('data already')


""""""""""" ~~~nets~~~ """""""""
start_epoch = 0
start_iteration = 0
model_rgb = ResNet101(BatchNorm=nn.BatchNorm2d, pretrained=bool(1-args.param), output_stride=16)
model_intergration = Integration()
model_att = TriAtt()


if args.param is True:
    model_rgb.load_state_dict(torch.load(os.path.join(args.snapshot_root, 'snapshot_iter_'+ parameters["snap_num"])))
    model_intergration.load_state_dict(torch.load(os.path.join(args.snapshot_root, 'integrate_snapshot_iter_'+ parameters["snap_num"])))
    model_att.load_state_dict(torch.load(os.path.join(args.snapshot_root, 'att_snapshot_iter_'+ parameters["snap_num"])))


if cuda:
    model_rgb = model_rgb.cuda()
    model_intergration = model_intergration.cuda()
    model_att = model_att.cuda()


if args.phase == 'train':

    #Trainer: class, defined in trainer.py
    optimizer_rgb = optim.SGD(model_rgb.parameters(), lr=cfg['lr'],momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])
    optimizer_inter = optim.SGD(model_intergration.parameters(), lr=cfg['lr'],momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])
    optimizer_att = optim.SGD(model_att.parameters(), lr=cfg['lr'],momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])
    training = Trainer(
        cuda=cuda,
        model_rgb=model_rgb,
        model_intergration=model_intergration,
        model_att=model_att,
        optimizer_rgb=optimizer_rgb,
        optimizer_inter=optimizer_inter,
        optimizer_att=optimizer_att,
        train_loader=train_loader,
        max_iter=cfg['max_iteration'],
        snapshot=cfg['spshot'],
        outpath=args.snapshot_root,
        sshow=cfg['sshow']
    )
    training.epoch = start_epoch
    training.iteration = start_iteration
    training.train()
else:
    res = []
    for id, (data, img_name, img_size) in enumerate(test_loader):
        print('testing bach %d' % (id+1))

        inputs = Variable(data).cuda()
        n, c, h, w = inputs.size()
        begin_time = time.time()

        low_1, low_2, high_1, high_2, high_3 = model_rgb(inputs)
        Features, _, _, Edge, _, _, Depth, Sal = model_intergration(low_1, low_2, high_1, high_2, high_3)
        outputs = model_att(Features, Edge, Sal, Depth)
        outputs = F.softmax(outputs, dim=1)
        outputs = outputs[0][1]
        outputs = outputs.cpu().data.resize_(h, w)
        end_time = time.time()
        run_time = end_time - begin_time
        res.append(run_time)
        imsave(os.path.join(MapRoot,img_name[0] + '.png'), outputs, img_size)

        # ---------------- Visual Results ------------------ #
        # Edge
        out1 = F.softmax(Edge, dim=1)
        out1 = out1[0][1]
        out1 = out1.cpu().data.resize_(h, w)
        imsave(os.path.join(args.out1, img_name[0] + '.png'), out1, img_size)
        # Depth  
        out2 = Depth[0][0]
        out2 = out2.cpu().data.resize_(h, w)
        imsave(os.path.join(args.out2, img_name[0] + '.png'), out2, img_size)
        # Sal-Att
        out3 = Sal[0][1]
        out3 = out3.cpu().data.resize_(h, w)
        imsave(os.path.join(args.out3, img_name[0] + '.png'), out3, img_size)
        # -------------------------------------------------- #

    print('The testing process has finished!')
    time_sum = 0
    for i in res:
       time_sum +=i
    print("FPS: %f"%(1.0/(time_sum/len(res))))
