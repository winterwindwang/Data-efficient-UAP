import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

import os

from torchvision import transforms, models
from torch.utils.data import DataLoader
from dataloader import RPC
from utils import *
from my_models import *
from gradcam import CAM_alex, CAM_resnet152, CAM_densenet
from functools import reduce
import torchvision.transforms.functional as TF
import argparse

import random 
random.seed(233)
torch.manual_seed(2333)
torch.cuda.manual_seed(2333)
np.random.seed(2333)

parser = argparse.ArgumentParser()
parser.add_argument("--lamb", type=float, default=1)
parser.add_argument('--model', type=str, default='resnet101')
parser.add_argument("--model_path", type=str)
parser.add_argument('--trans', type=str, default='dtcr')
parser.add_argument('--patch_raw', type=str, default='')

parser.add_argument('--data_path', type=str)
parser.add_argument('--train_file', type=str)

args = parser.parse_args()

lamb = args.lamb

def make_log_dir():
    logs = {
        'attack': args.model,
        'epoch': 5,
        'lr': 0.01,
        'loss': 'ssim',
        'trans': 'affine_' + args.trans,
        'lambda': lamb,
        # 'dataset': '0_100',
        'patch_raw': args.patch_raw.split('/')[-1][:-4],
    }
    dir_name = ""
    for key in logs.keys():
        dir_name += str(key) + '-' + str(logs[key]) + '+'
    dir_name = 'logs/' + dir_name
    print(dir_name)
    if not (os.path.exists(dir_name)):
        os.makedirs(dir_name)
    return dir_name

log_dir = make_log_dir()
    
pic_path = args.data_path
train_info= args.train_file


preprocess = transforms.Compose([
                transforms.Resize((32, 32)),
                #transforms.CenterCrop(224),
                # transforms.ToTensor(),
                transforms.Normalize(mean = [0.5,0.5,0.5], std = [0.5,0.5,0.5])
            ])

def loss_attack(preds, label):
    label = label[0]
    _, idx = torch.sort(preds, dim=1, descending=True)
    idx = idx[0]
    # print(label)
    if idx[0] == label:
        return preds[0][label] - preds[0][idx[1]]
    else:
        return preds[0][label] - preds[0][idx[0]]

cam_edge = 16

vis = np.zeros((cam_edge, cam_edge))

def dfs(x1, x, y, points):
    points.append(x1[x][y])
    global vis
    vis[x][y] = 1
    n = 1
    # print(x, y)
    if x+1 < cam_edge and x1[x+1][y] > 0 and not  vis[x+1][y]:
        n += dfs(x1, x+1, y, points)
    if x-1 >= 0 and x1[x-1][y] > 0 and not  vis[x-1][y]:
        n += dfs(x1, x-1, y, points)
    if y+1 < cam_edge and x1[x][y+1] > 0 and not  vis[x][y+1]:
        n += dfs(x1, x, y+1, points)
    if y-1 >= 0 and x1[x][y-1] > 0 and not  vis[x][y-1]:
        n += dfs(x1, x, y-1, points)
    return n
    
def loss_midu(x1):
    # print(torch.gt(x1, torch.ones_like(x1) * 0.1).float())
    
    x1 = torch.tanh(x1)
    global cam_edge
    cam_edge = x1.shape[1]
    # print(cam_edge)
    global vis
    vis = np.zeros((cam_edge, cam_edge))
    
    loss = []
    # print(x1)
    for i in range(cam_edge):
        for j in range(cam_edge):
            if x1[i][j] > 0 and not vis[i][j]:
                point = []
                n = dfs(x1, i, j, point)
                # print(n)
                # print(point)
                loss.append( reduce(lambda x, y: x + y, point) / (cam_edge * cam_edge + 1 - n) )
    # print(vis)
    if len(loss) == 0:
        return torch.zeros(1).cuda()
    return reduce(lambda x, y: x + y, loss) / len(loss)

def loss_std(x1):
    return torch.std(x1)

def loss_mse(x1):
    # print(x1.shape)
    
    ret = torch.sum(torch.pow(x1, 2)) / (x1.shape[0] * 256)
    return ret

ssim = SSIM()
def loss_compare(x, x_raw):
    # print(x.shape)
    x = x.unsqueeze(0).unsqueeze(0)
    x_raw = x_raw.unsqueeze(0).unsqueeze(0)
    # print(x.shape)
    return ssim(x, x_raw)


_d = (-30, 30) if 'd' in args.trans else (0, 0)
_t = (0.05, 0.05) if 't' in args.trans else (0.0, 0.0)
_c = (0.8, 1.2) if 'c' in args.trans else (1.0, 1.0)
_r = (-20, 20) if 'r' in args.trans else (0, 0)

print('AFFINE:', _d, _t, _c, _r)
def affine(img, mask):
    '''
    degree = random.randint(-3, 3) * 10
    
    shift_x = random.randint(-2, 2) * 5
    shift_y = random.randint(-2, 2) * 5
    
    scale = random.randint(-2, 2) * 0.1 + 1
    
    shear_x = random.randint(-3, 3) * 10
    shear_y = random.randint(-3, 3) * 10
    '''

    
    degree, (shift_x, shift_y), scale, (shear_x, shear_y) = transforms.RandomAffine.get_params(_d, _t, _c, _r, (512, 512))
    
    img = TF.affine(img, angle=degree, translate=(shift_x, shift_y), scale=scale, shear=(shear_x, shear_y))
    mask = TF.affine(mask, angle=degree, translate=(shift_x, shift_y), scale=scale, shear=(shear_x, shear_y))
    
    return img, mask
    
def perspective(img, mask):
    startpoint, endpoint = transforms.RandomPerspective.get_params(512, 512, 0.5)
    
    img = TF.perspective(img, startpoints=startpoint, endpoints=endpoint)
    mask = TF.perspective(mask, startpoints=startpoint, endpoints=endpoint)
    
    return img, mask
    
    

def gen_patch(EPOCH=1, LR=0.01, BATCH_SIZE=1):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.model=='alex':
        model = get_alexnet(path=args.model_path)
        gradcam = CAM_alex(model, log_dir) 
       
    elif args.model=='vgg16':
        model = get_vgg16(path=args.model_path)
        gradcam = CAM_alex(model, log_dir) 
        
    elif args.model=='resnet101':
        model = get_resnet101(path=args.model_path)
        gradcam = CAM_resnet152(model, log_dir)
        
    elif args.model=='resnet152':
        model = get_resnet152(path=args.model_path)
        gradcam = CAM_resnet152(model, log_dir)
        
    elif args.model=='densenet161':
        model = get_densenet161(path=args.model_path)
        gradcam = CAM_densenet(model, log_dir)
        
    else:
        print("ERROR MODEL!")
        return 
    # print(model)
    model.to(device)
    
    if args.patch_raw != '':
        adv_patch = torch.load(args.patch_raw)
     
    
    labelset, img_path, centerset, bboxset, _ = get_pic_info(pic_path, train_info)
    train_dataset = RPC(labelset, img_path, centerset, bboxset, pic_size=512)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    
    
    
    adv_patch = preprocess(adv_patch).to(device)
    adv_patch = torch.autograd.Variable(adv_patch, requires_grad=True)
    tensor_show(adv_patch.cpu(), os.path.join(log_dir, 'patch_raw.jpg'))
    
    optimizer = torch.optim.Adam([adv_patch], lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1, 7, 15, 25, 60, 120], gamma=1/3)
    
    # cirterion = nn.CrossEntropyLoss()
    
    model.eval()
    model_best = 1.0
    best_epoch = 0
    iters = 0
    for _ in range(EPOCH):
        print("Epoch:", _)
        print(optimizer)

        correct = 0
        total = 0
        total_loss = 0
        tqdm_loader = tqdm(train_loader)
        for i, data in enumerate(tqdm_loader):
            iters += 1
            
            inputs, labels, path, center = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            
            # print(center)
           
            
            patch, mask = pad_transform(adv_patch, 512, 32, center)
            patch, mask = affine(patch, mask)
            # patch, mask = perspective(patch, mask)
            
            inputs.requires_grad_(True)
            cam_raw, preds_raw = gradcam(inputs, labels[0], iters)
            # inputs.requires_grad_(False)
            inputs = inputs.detach()
            cam_raw = cam_raw.detach()
            
            adv_inputs = inputs * (1 - mask) + mask * patch
            if iters % 10 == 0:
                tensor_show(patch.cpu(), os.path.join(log_dir, 'patch_trans.jpg'))
                tensor_show(adv_inputs[0].cpu(), os.path.join(log_dir, 'adv_input.jpg'))
            
            cam, preds = gradcam(adv_inputs, labels[0], iters)
            # preds = F.softmax(outputs, dim=1)
            # print(outputs[0])
            # print(outputs.shape)
            loss1 = loss_attack(preds, labels)
            loss2 = loss_compare(cam, cam_raw) # loss_midu(cam)
            
            loss = loss1 + lamb * loss2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            adv_patch.data.clamp_(-1,1)
            
            total += labels.size(0)
            total_loss += loss.sum()

            tqdm_loader.set_description('loss: %.6f, loss1: %.6f, loss2: %.6f' % (total_loss / total, loss1, loss2))
            
        total_loss /= total
        with open(os.path.join(log_dir, 'train_log.txt'), 'a') as f:
            f.write('epoch: %d, loss: %.6f\n' % (_, total_loss))
        
        tensor_show(adv_patch.cpu(), os.path.join(log_dir, 'adv_patch_%d.jpg' % _))
        torch.save(adv_patch.cpu(), os.path.join(log_dir, 'adv_patch_%d.pkl' % _))
        
      
        scheduler.step()
    
    
if __name__ == '__main__':
    gen_patch()