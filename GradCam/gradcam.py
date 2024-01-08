import argparse
import cv2
import numpy as np
import torch
from torch.autograd import Function
import torch.nn.functional as F
from torchvision import models
from PIL import Image
import os
from my_models import *
from torchvision import transforms
import torchvision.transforms.functional as TF
import argparse



class GradCam():
    
    hook_a, hook_g = None, None
    
    hook_handles = []
    
    def __init__(self, model, conv_layer, use_cuda=True):
        
        self.model = model.eval()
        self.use_cuda=use_cuda
        if self.use_cuda:
            self.model.cuda()
        
        self.hook_handles.append(self.model._modules.get(conv_layer).register_forward_hook(self._hook_a))
        
        self._relu = True
        self._score_uesd = True
        self.hook_handles.append(self.model._modules.get(conv_layer).register_backward_hook(self._hook_g))
        
    
    def _hook_a(self, module, input, output):
        self.hook_a = output
        
    def clear_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
    
    def _hook_g(self, module, grad_in, grad_out):
        # print(grad_in[0].shape)
        # print(grad_out[0].shape)
        self.hook_g = grad_out[0]
    
    def _backprop(self, scores, class_idx):
        
        loss = scores[:, class_idx].sum() # .requires_grad_(True)
        self.model.zero_grad()
        loss.backward(retain_graph=True)
        
    def _get_weights(self, class_idx, scores):
        """Computes the weight coefficients of the hooked activation maps"""

        # Backpropagate
        self._backprop(scores, class_idx)
        # Global average pool the gradients over spatial dimensions
        return self.hook_g.squeeze(0).mean(axis=(1, 2))
    '''
    def _get_weights(self, class_idx, scores):
        
        self._backprop(scores, class_idx)
        
        grad_2 = self.hook_g.pow(2)
        grad_3 = self.hook_g.pow(3)
        alpha = grad_2 / (1e-13 + 2 * grad_2 + (grad_3 * self.hook_a).sum(axis=(2, 3), keepdims=True))

        # Apply pixel coefficient in each weight
        return alpha.squeeze_(0).mul_(torch.relu(self.hook_g.squeeze(0))).sum(axis=(1, 2))
    '''
    def __call__(self, input, class_idx=None):
        # print(input.shape)
        # if self.use_cuda:
        #     input = input.cuda()
        scores = self.model(input)
        if class_idx == None:
            class_idx = int(torch.argmax(scores))
            print(class_idx)
        pred = F.softmax(scores, dim=1)
        # print(scores)
        weights = self._get_weights(class_idx, scores)
        # print(input.grad)
        # rint(weights)
        cam = (weights.unsqueeze(-1).unsqueeze(-1) * self.hook_a.squeeze(0)).sum(dim=0)
        
        # print(cam.shape)
        # self.clear_hooks()
        cam_np = cam.data.cpu().numpy()
        cam_np = np.maximum(cam_np, 0)
        cam_np = cv2.resize(cam_np, input.shape[2:])
        cam_np = cam_np - np.min(cam_np)
        cam_np = cam_np / np.max(cam_np)
        return cam, cam_np, pred

class CAM_alex:
    
    def __init__(self, model, log_dir='./'):
        # model = models.get_alexnet(pretrained=True)
        self.grad_cam = GradCam(model=model, conv_layer='features', use_cuda=True)
        self.log_dir = log_dir
        
    def __call__(self, img, index=None, iters=0):

        ret, mask, pred = self.grad_cam(img, index)

        # print(img.shape)
        if iters % 10 == 0:
            self.show_cam_on_image(img, mask)
        return ret, pred

    def show_cam_on_image(self, img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)[:, :, ::-1]
        heatmap = np.float32(heatmap) / 255
        cam_pure = heatmap
        cam_pure = cam_pure / np.max(cam_pure)
        img = img.data.cpu().numpy()[0].transpose((1, 2, 0))
        img = img * 0.5 + 0.5
        cam = np.float32(img) + heatmap
        cam = cam / np.max(cam)
        
        Image.fromarray(np.uint8(255 * cam)).save(os.path.join(self.log_dir, 'cam.jpg'))
        Image.fromarray(np.uint8(255 * mask)).save(os.path.join(self.log_dir, 'cam_b.jpg'))
        
        Image.fromarray(np.uint8(255 * cam_pure)).save(os.path.join(self.log_dir, 'cam_p.jpg'))
            
        # cv2.imwrite("cam.jpg", np.uint8(255 * cam))
        
class CAM_resnet152:
    
    def __init__(self, model, log_dir='./'):
        # model = models.get_alexnet(pretrained=True)
        self.grad_cam = GradCam(model=model, conv_layer='layer4', use_cuda=True)
        self.log_dir = log_dir
        
    def __call__(self, img, index=None, iters=0):
        img.requires_grad_(True)
        ret, mask, pred = self.grad_cam(img, index)

        # print(img.shape)
        if iters % 1 == 0:
            self.show_cam_on_image(img, mask)
        return ret, pred

    def show_cam_on_image(self, img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)[:, :, ::-1]
        heatmap = np.float32(heatmap) / 255
        cam_pure = heatmap
        cam_pure = cam_pure / np.max(cam_pure)
        img = img.data.cpu().numpy()[0].transpose((1, 2, 0))
        img = img * 0.5 + 0.5
        cam = np.float32(img) + 0.5 * heatmap
        cam = cam / np.max(cam)
        
        Image.fromarray(np.uint8(255 * img)).save(os.path.join(self.log_dir, 'raw.jpg'))
        Image.fromarray(np.uint8(255 * cam)).save(os.path.join(self.log_dir, 'cam.jpg'))
        Image.fromarray(np.uint8(255 * mask)).save(os.path.join(self.log_dir, 'cam_b.jpg'))
        
        Image.fromarray(np.uint8(255 * (0.3 + 0.7 * cam_pure))).save(os.path.join(self.log_dir, 'cam_p.jpg'))
            
        # cv2.imwrite("cam.jpg", np.uint8(255 * cam))

class CAM_densenet:
    
    def __init__(self, model, log_dir='./'):
        # model = models.get_alexnet(pretrained=True)
        self.grad_cam = GradCam(model=model, conv_layer='features', use_cuda=True)
        self.log_dir = log_dir
        
    def __call__(self, img, index=None, iters=0):

        ret, mask, pred = self.grad_cam(img, index)

        # print(img.shape)
        if iters % 10 == 0:
            self.show_cam_on_image(img, mask)
        return ret, pred

    def show_cam_on_image(self, img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)[:, :, ::-1]
        heatmap = np.float32(heatmap) / 255
        cam_pure = heatmap
        cam_pure = cam_pure / np.max(cam_pure)
        img = img.data.cpu().numpy()[0].transpose((1, 2, 0))
        img = img * 0.5 + 0.5
        cam = np.float32(img) + heatmap
        cam = cam / np.max(cam)
        
        Image.fromarray(np.uint8(255 * cam)).save(os.path.join(self.log_dir, 'cam.jpg'))
        Image.fromarray(np.uint8(255 * mask)).save(os.path.join(self.log_dir, 'cam_b.jpg'))
        
        Image.fromarray(np.uint8(255 * cam_pure)).save(os.path.join(self.log_dir, 'cam_p.jpg'))
            
        # cv2.imwrite("cam.jpg", np.uint8(255 * cam))
        
preprocess = transforms.Compose([
                transforms.Resize((512, 512)),
                #transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean = [0.5,0.5,0.5], std = [0.5,0.5,0.5])
            ])

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model", type=str, default='resnet101')
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--input", type=str)
    parser.add_argument('--output', type=str, default='patch_raw.pkl')
    
    args = parser.parse_args()
    
    img_path = args.input 
    class_index = None
    
    if args.model=='alex':
        model = get_alexnet(path=args.model_path)
        grad_cam = GradCam(model=model, conv_layer='features', use_cuda=True)
        
    elif args.model=='vgg16':
        model = get_vgg16(path=args.model_path)
        grad_cam = GradCam(model=model, conv_layer='features', use_cuda=True)
        
    elif args.model=='resnet101':
        model = get_resnet101(path=args.model_path)
        grad_cam = GradCam(model=model, conv_layer='layer4', use_cuda=True)
        
    elif args.model=='densenet161':
        model = get_densenet161(path=args.model_path)
        grad_cam = GradCam(model=model, conv_layer='features', use_cuda=True)
    
    
    
    
    # img = Image.open(img_path).convert('RGB')
    img = torch.load(img_path)
    
    
    # img_inpug = preprocess(img).unsqueeze(0).cuda()
    img_inpug = img.unsqueeze(0).cuda()
    img_inpug.requires_grad_(True)
    ret, mask, pred = grad_cam(img_inpug, class_index)
    
    
    img_np = img.data.cpu().numpy() / 255
    img_np = img_np.transpose((1, 2, 0))
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)[:, :, ::-1]
    heatmap = np.float32(heatmap) / 255
    cam = 0.5 * np.float32(img_np) + 0.5 * heatmap
    cam = cam / np.max(cam)
    
    # print(cam.shape)
    Image.fromarray(np.uint8(255 * cam)).save('cam.jpg')
    # print(model)
    
    max_num = 0
    max_i, max_j = 0, 0
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i][j] > max_num:
                max_num = mask[i][j]
                max_i, max_j = i, j
    
    # print(img.shape)
    # print(max_i, max_j)
    if max_i < 16:
        max_i = 16
    if max_j < 16:
        max_j = 16
        
    img_crop = TF.crop(img, max_i - 16, max_j - 16, 32, 32)
    # print(img_crop.shape)
    torch.save(img_crop, args.output) 
            