import torch
import torch.nn as nn

from torchvision import models

def get_alexnet(finetune=False, path=None):
    model = models.alexnet(pretrained=True)
    # print(model)
    '''
    model.classifier = nn.Sequential(
        nn.Linear(256 * 6 * 6, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 200),
    )
    '''
    model.classifier[6] = nn.Linear(4096, 200)
    # print(model)
    for param in model.parameters():
            param.requires_grad = False
    if finetune:
        for param in model.classifier.parameters():
            param.requires_grad = True
    if path != None:
        model.load_state_dict(torch.load(path))
    
    return model

def get_vgg16(finetune=False, path=None):
    model = models.vgg16_bn(pretrained=True)
    
    model.classifier[6] = nn.Linear(4096, 200)
    for param in model.parameters():
            param.requires_grad = False
    if finetune:
        for param in model.classifier.parameters():
            param.requires_grad = True
    # print(model)
    
    if path != None:
        model.load_state_dict(torch.load(path))
    return model

def get_resnet101(finetune=False, path=None):
    model = models.resnet101(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 200)
    for param in model.parameters():
            param.requires_grad = False
    if finetune:
        for param in model.fc.parameters():
            param.requires_grad = True
    if path != None:
        model.load_state_dict(torch.load(path))
    return model
    
def get_resnet152(finetune=False, path=None):
    model = models.resnet152(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 200)
    for param in model.parameters():
            param.requires_grad = False
    if finetune:     
        for param in model.fc.parameters():
            param.requires_grad = True
    if path != None:
        model.load_state_dict(torch.load(path))
    return model

def get_densenet161(finetune=False, path=None):
    model = models.densenet161(pretrained=True)
    model.classifier = nn.Linear(model.classifier.in_features, 200)
    for param in model.parameters():
            param.requires_grad = False
    if finetune:     
        for param in model.parameters():
            param.requires_grad = True
    if path != None:
        model.load_state_dict(torch.load(path))
    return model


if __name__ == '__main__':
    model = get_densenet161()
    print(model)