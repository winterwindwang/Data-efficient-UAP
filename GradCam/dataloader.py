#!/usr/bin/env python
# coding: utf-8

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torch.nn import Parameter
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable
from PIL import Image
import math
from torchvision import models
from collections import OrderedDict
from collections import namedtuple
#from utils import OHE_labels
import pickle
import numpy as np
import os

    
class RPC(torch.utils.data.Dataset): 
    def __init__(self, labelset, img_path, centerset, bboxset, pic_size, lineset=None): 
        self.img_path = img_path
        self.labelset = labelset
        self.centerset = centerset
        self.bboxset = bboxset
        self.pic_size = pic_size
        self.lineset = lineset
        
    def __getitem__(self, index):     
        # index = 0
        
        
        image = Image.open(self.img_path[index]).convert('RGB')
       
        image = self.transform(image)
        
        path = self.img_path[index]
        label = self.labelset[index]
        
        raw_size = (2592, 1944)
        center = self.centerset[index]
        center = (center[0] / raw_size[0], center[1] / raw_size[1])
        center = (int(center[0] * 512), int(center[1] * 512))
        
        
        if self.lineset != None:
            return image, label, path, center, self.lineset[index]
        else:
            return image, label, path, center
    
    def __len__(self):
        return len(self.img_path)

    def transform(self, image):
        preprocess = transforms.Compose([
            transforms.Resize((self.pic_size,self.pic_size)),
            #transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.5,0.5,0.5],std = [0.5,0.5,0.5])
        ])
        image = preprocess(image)
        return image

