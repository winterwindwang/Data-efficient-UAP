import torch
from torchvision import transforms
import os
import time
import json
import numpy as np
import random
from torchvision import models


class Normalize(torch.nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, tensor):
        return normalize_fn(tensor, self.mean, self.std)

    def backward(self):
        return "mean={}, std={}".format(self.mean, self.std)

def normalize_fn(tensor, mean, std):
    """
    Differentiable version of torchvision.functional.normalize
    - default assumes color channel is at dim = 1
    """
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std)

def get_model(args):
    if "resnet50" in args.model_name:
        model = models.resnet50(pretrained=True)
        args.eps_step = 0.001
    elif "resnet101" in args.model_name:
        model = models.resnet101(pretrained=True)
        args.eps_step = 0.001
    elif "resnet152" in args.model_name:
        model = models.resnet152(pretrained=True)
        args.eps_step = 0.001
    elif "resnext50" in args.model_name:
        model = models.resnext50_32x4d(pretrained=True)
        args.eps_step = 0.005
    elif "wideresnet" in args.model_name:
        model = models.wide_resnet50_2(pretrained=True)
        args.eps_step = 0.005
    elif "vgg16" in args.model_name:
        model = models.vgg16(pretrained=True)
        args.eps_step = 0.003
    elif "vgg19" in args.model_name:
        model = models.vgg19(pretrained=True)
        args.eps_step = 0.003
    elif "densenet121" in args.model_name:
        model = models.densenet121(pretrained=True)
        args.eps_step = 0.01
    elif "densenet161" in args.model_name:
        model = models.densenet161(pretrained=True)
        args.eps_step = 0.005
    elif "inception_v3" in args.model_name:
        model = models.inception_v3(pretrained=True)
        args.eps_step = 0.003
    elif "googlenet" in args.model_name:
        model = models.googlenet(pretrained=True)
        args.eps_step = 0.005
    elif "alexnet" in args.model_name:
        model = models.alexnet(pretrained=True)
        args.eps_step = 0.001
    elif "mnasnet10" in args.model_name:
        model = models.mnasnet1_0(pretrained=True)
        args.eps_step = 0.001
    elif "efficientnetb0" in args.model_name:
        model = models.efficientnet_b0(pretrained=True)
        args.eps_step = 0.001

    normalize = Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
    model = torch.nn.Sequential(normalize, model)
    return model, args
