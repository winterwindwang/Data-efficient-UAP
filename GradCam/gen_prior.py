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
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--lamb", type=float, default=1)
parser.add_argument('--model', type=str, default='resnet101')
parser.add_argument('--model_path', type=str)
parser.add_argument('--wrong_file', type=str)
parser.add_argument('--data_path', type=str)
args = parser.parse_args()
    
def make_log_dir():
    logs = {
        'prior': args.model+'_wrong',
        'epoch': 50,
        'lr': 0.01,
        'lambda': args.lamb
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


class FeatureExtractor():
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.features = []
        self.hook_handles = []
        if 'resnet' in args.model:
            self.hook_handles.append(self.model.layer4.register_forward_hook(self._hook_f))
        else:
            self.hook_handles.append(self.model.features.register_forward_hook(self._hook_f))
        
    def _hook_f(self, module, input, output):
        self.features.append(output)
        
    def run(self, _input):
        self.features = []
        output = self.model(_input)
        return output, self.features
        

def gen_prior(EPOCH=50, LR=0.01, epsilon=1):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.model=='alex':
        model = get_alexnet(path=args.model_path)
        
    elif args.model=='vgg16':
        model = get_vgg16(path=args.model_path)
        
    elif args.model=='resnet101':
        model = get_resnet101(path=args.model_path)
        
    elif args.model=='resnet152':
        model = get_resnet152(path=args.model_path)
        
    elif args.model=='densenet161':
        model = get_densenet161(path=args.model_path)
        
    else:
        print("ERROR MODEL!")
        return 
    
    pic_info = args.wrong_file
    
    model.to(device)
    print(model)
    feature_extractor = FeatureExtractor(model)
        
    labelset, img_path, centerset, bboxset, lineset = get_pic_info(pic_path, pic_info)
    dataset = RPC(labelset, img_path, centerset, bboxset, pic_size=512)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=8)
    
    prior = dataset[0][0]
    prior.requires_grad_(True)
    tensor_show(prior, os.path.join(log_dir, 'raw_prior.jpg'))
    
    optimizer = torch.optim.Adam([prior], lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 8, 15, 25, 50], gamma=1/3)
    
    criterion_MSE = nn.MSELoss()
    
    best =10000000000
    best_epoch = 0
    
    for _ in range(EPOCH):
        print("EPOCH", _)
        print(optimizer)
        
        total = 0
        correct = 0
        total_loss = 0
        tqdm_loader = tqdm(loader)
        for i, data in enumerate(tqdm_loader):
            inputs, labels, path, center = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            b = labels.size(0)
            
            output, features = feature_extractor.run(inputs)
            # print(features[0].shape)
            output_p, features_p = feature_extractor.run(prior.unsqueeze(0).cuda())
            # print(features_p[0].shape)
                        
            output_p = F.softmax(output_p, dim=1)


            loss1 = torch.sum(torch.mul(output_p, torch.log(output_p + 1e-13)))
            
            gram_p = gram_matrix(features_p[0])
            gram_p = gram_p.repeat(b, 1, 1)
           
            
            gram_img = gram_matrix(features[0])
            

            loss2 = criterion_MSE(gram_p, gram_img)
            
            # print(loss2)
            if args.lamb == -1:
                loss = loss2
            else:
                loss = args.lamb * loss1 + loss2
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            prior.data.clamp_(-1, 1)
            
            total += b
            total_loss += loss
            tqdm_loader.set_description('total_loss: %.4f, loss: %.4f, loss1: %.4f, loss2: %.4f' % (total_loss / total, loss, loss1, loss2))
        
        
        
        if _ % 5 == 0:
            torch.save(prior, os.path.join(log_dir, 'prior_%03d.pkl' % _))
            tensor_show(prior, os.path.join(log_dir, 'prior_%03d.jpg' % _))
        
        total_loss /= total
        if total_loss < best:
            best = total_loss
            best_epoch = _
            torch.save(prior, os.path.join(log_dir, 'prior_best.pkl'))
            tensor_show(prior, os.path.join(log_dir, 'prior_best.jpg'))
        
        tensor_show(prior, os.path.join(log_dir, 'prior.jpg'))
        
        with open(os.path.join(log_dir, 'train_log.txt'), 'a') as f:
            f.write('epoch: %d, total_loss: %.8f, best: %d\n' % (_, total_loss, best_epoch))
            
        scheduler.step()
    
    
if __name__ == '__main__':
    gen_prior()
