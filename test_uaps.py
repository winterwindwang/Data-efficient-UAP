import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
import argparse
import sys
from torchvision.datasets import ImageFolder


parser = argparse.ArgumentParser(description=" Data efficient universal adversarial perturbations")
parser.add_argument('--data_dir', type=str, default='./data/', help="The imagenet folder(contains: train folder and validation folder)")
parser.add_argument('--model_name', type=str, default='vgg19')
parser.add_argument('--checkpoints', type=str, default='checkpoints/')
parser.add_argument('--batch_size', type=int, default=32)


class Normalizer(nn.Module):
    def __init__(self, mean, std):
        super(Normalizer, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)


def normalize_fn(tensor, mean, std):
    """
    Differentiable version of torchvision.functional.normalize
    - default assumes color channel is at dim = 1
    """
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std)


@torch.no_grad()
def evaluate_src_attack(args, val_dataset, uap, **kwargs):
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,  shuffle=False, num_workers=1, pin_memory=True)
    model = pytorch_models[args.model_name](pretrained=True)
    model = nn.Sequential(normalize, model)
    model.eval()
    model = model.to(device)

    correct = 0
    ori_correct = 0
    fool_num = 0
    n = 0

    for i, (images, labels) in enumerate(val_loader):
        images = images.to(device)
        labels = labels.to(device)
        ori_output = model(images)
        ori_pred = torch.argmax(ori_output, dim=1)

        pred_ori_idx = labels == ori_pred
        ori_correct += pred_ori_idx.sum().item()

        adv_img = torch.clamp((images + uap.repeat([images.size(0), 1, 1, 1]).to(device)), 0, 1)
        output = model(adv_img)
        pred = torch.argmax(output, dim=1)

        pred_pert_idx = labels == pred

        correct += (pred_pert_idx ^ pred_ori_idx).sum().item()

        fool_num += (ori_pred != pred).sum().item()

        n += images.size(0)
    print("total:{}, success pred: {}, success attack: {}".format(n, ori_correct, correct))
    return np.round(100 * (ori_correct / n), 2), np.round(100 * (correct / n), 2), np.round(
        100 * (fool_num / n), 2)

def evaluate_both_models(args, uap_dict):
    model_names = ['resnet50', 'resnet152', 'vgg16', 'vgg19', 'densenet121', 'densenet161', 'alexnet']
    if not isinstance(uap_dict, dict):
        uap_dict = [uap_dict]

    for key, value in uap_dict.items():
        args.perturbation = value
        curr_acc = []
        curr_asr = []
        curr_fool_rate = []
        for model_name in model_names:
            TEST_TRANSFORMS = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor()
            ])
            uap = torch.load(args.perturbation)
            if len(uap.size()) == 4:
                pass
            else:
                uap = uap.unsqueeze(0)

            args.model_name = model_name
            # dataset
            valdir = os.path.join(args.data_dir, 'val')
            val_dataset = ImageFolder(root=valdir, transform=TEST_TRANSFORMS)
            acc, asr, fool_rate = evaluate_src_attack(args, val_dataset, uap, test_trans=TEST_TRANSFORMS)
            curr_acc.append(acc)
            curr_asr.append(asr)
            curr_fool_rate.append(fool_rate)
        print(f"UAP method: {key}_{value}")
        print(f"UAP: {args.perturbation}")
        print("model", model_names)
        print('acc:', curr_acc)
        print('asr:', curr_asr)
        print('fool_rate:', curr_fool_rate)
        print("\n")


def check_files(uap_dict):
    for key, value in uap_dict.items():
        if not os.path.exists(value):
            sys.exit(f"file {value} not exit")


if __name__ == "__main__":
    args = parser.parse_args()
    path = './perturbations/'
    pytorch_models = {
        'alexnet': models.alexnet,
        'vgg16': models.vgg16,
        'vgg19': models.vgg19,
        'densenet121': models.densenet121,
        'densenet161': models.densenet161,
        'resnet50':models.resnet50,
        'resnet152':models.resnet152,
    }

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    IMGNET_MEAN = [0.485, 0.456, 0.406]
    IMGNET_STD = [0.229, 0.224, 0.225]
    normalize = Normalizer(mean=IMGNET_MEAN, std=IMGNET_STD)
    input_size = 224
    uap_files = {
        "our_uap_resnet50": f"{path}/our_uap_resnet50_eps10_eta0001_fr9126.pth",
        "our_uap_resnet152": f"{path}/our_uap_resnet152_eps10_eta0001_fr8875.pth",
        "our_uap_vgg16": f"{path}/our_uap_vgg16_eps10_eta0001_fr9565.pth",
        "our_uap_vgg19": f"{path}/our_uap_vgg19_eps10_eta0001_fr9421.pth",
        "our_uap_densenet121": f"{path}/our_uap_densenet121_eps10_eta0001_fr9125.pth",
        "our_uap_densenet161": f"{path}/our_uap_densenet161_eps10_eta0001_fr9413.pth",
        "our_uap_alexnet": f"{path}/our_uap_alexnet_eps10_eta0001_fr932.pth",
    }
    check_files(uap_files)
    print("Check done!!！")
    evaluate_both_models(args, uap_files)
