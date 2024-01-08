import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
import argparse
from PIL import Image
import torch.nn.functional as F
import sys
from torchvision.datasets import ImageFolder
from data_loader import ImageNet
from tqdm import tqdm
import config


parser = argparse.ArgumentParser(description=" transfer_learning")
parser.add_argument('--is_downstream', type=bool, default=False, help='Whether to evaulate the downstream task')
parser.add_argument('--data_path', type=str, default='xxx/ImageNet/ILSVRC2012_img_val')
parser.add_argument('--model_name', type=str, default='densenet161')
parser.add_argument('--downstream_model_name', choices=["densenet121", "resnet50", "resnet101", "resnet152", "vgg19", "resnext50", "wideresnet50"], default='resnet50')
parser.add_argument('--checkpoints', type=str, default='checkpoints/')
parser.add_argument('--downstream_ckpt', type=str, default='checkpoints/')
parser.add_argument('--finetune_type', choices=["fixed", "fullnet"], default='fixed', help="Choice the finetuned type for downstream task")
parser.add_argument('--type', type=str, default='fullnet')
parser.add_argument('--perturbation', type=str, default='', help='perturbation file path')
parser.add_argument('--dataset', type=str, default="flowers", help="['birdsnap', 'caltech101', 'caltech256', 'dtd', 'fgvc_aircraft_2013b', 'flowers', 'food', 'pet', 'stanford_cars', 'SUN397']")
parser.add_argument('--num_classes', type=int, default=100)
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

def process_model(model_name, model, num_classes, additional_hidden=0, mode='fullnet'):
    if mode == 'fixed':
        for param in model.parameters():
            param.requires_grad=False
    if model_name in ["resnet", "resnet18", "resnet50", "wide_resnet50_2", "wide_resnet50_4", "resnext50_32x4d", 'shufflenet']:
        num_ftrs = model.fc.in_features
        if additional_hidden == 0:
            model.fc = torch.nn.Linear(num_ftrs, num_classes)
        else:
            model.fc = torch.nn.Sequential(
                *list(sum([[nn.Linear(num_ftrs, num_ftrs), nn.ReLU()] for i in range(additional_hidden)], [])),
                nn.Linear(num_ftrs, num_classes)
             )
        input_size = 224
    elif 'vgg' in model_name:
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif 'inception' in model_name:
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299
    elif 'densenet' in model_name:
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    else:
        ValueError("Invalid model type, exiting...")
    return model, input_size

@torch.no_grad()
def evaluate_attack_downstream_task(args, val_dataset, uap):
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,  shuffle=False, drop_last=True)
    model = pytorch_models[args.model_name](pretrained=True)
    model, input_size = process_model(args.model_name, model, val_dataset.num_classes, mode=args.finetune_type)
    ckpt_dict = load_downstream_model(args)
    ckpt = torch.load(ckpt_dict[args.dataset])
    model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()
    correct = 0
    ori_correct = 0
    n = 0
    phar = tqdm(val_loader, desc="loader")
    for i, (images, labels) in enumerate(phar):
        images = images.to(device)
        labels = labels.to(device)
        ori_output = model(images)
        ori_pred = torch.argmax(ori_output, dim=1)
        ori_correct += (labels == ori_pred).sum()

        adv_img = torch.clamp((images + uap.to(device)), 0, 1)
        output = model(adv_img)
        pred = torch.argmax(output, dim=1)
        correct += (labels != pred).sum()
        n += images.size(0)
        phar.set_description("attack success:{}, accuracy:{}".format(100 * (correct / n), 100 * (ori_correct / n)))
    print("total:{}, success pred: {}, success attack: {}".format(val_dataset.__len__(), ori_correct, correct))


@torch.no_grad()
def evaluate_src_attack(args, val_dataset, uap, **kwargs):
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,  shuffle=False, num_workers=8, pin_memory=True)

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

        correct += (pred_pert_idx & pred_ori_idx).sum().item()

        fool_num += (ori_pred != pred).sum().item()

        n += images.size(0)
    print("total:{}, success pred: {}, success attack: {}".format(n, ori_correct, correct))
    return np.round(100 * (ori_correct / n), 2), np.round(100 * (correct / n), 2), np.round(
        100 * (fool_num / n), 2)

def evaluate_both_models(args, uap_dict):
    model_names = ['vgg16', 'vgg19', 'resnet50', 'resnet101', 'resnet152', 'resnext50', 'wideresnet', 'efficientnetb0', 'densenet121', 'densenet161', 'googlenet','mnasnet']
    if not isinstance(uap_dict, dict):
        uap_dict = [uap_dict]

    for key, value in uap_dict.items():
        args.perturbation = value
        curr_acc = []
        curr_asr = []
        curr_fool_rate = []
        for model_name in model_names:
            if 'inception_v3' in model_name:
                TRAIN_TRANSFORMS = transforms.Compose([
                    transforms.RandomResizedCrop(299),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ])
                TEST_TRANSFORMS = transforms.Compose([
                    transforms.Resize(330),
                    transforms.CenterCrop(299),
                    transforms.ToTensor()
                ])
                if args.perturbation.endswith('.pth'):
                    uap = torch.load(args.perturbation)
                elif args.perturbation.endswith('.tar'):
                    ckpt = torch.load(args.perturbation)
                    try:
                        uap = ckpt['state_dict']['uap']
                    except:
                        uap = ckpt['uap']
                else:
                    uap = Image.open(args.perturbation).convert("RGB")
                    uap = np.asarray(uap) / 255
                    uap = np.transpose(uap, (2, 0, 1))
                    uap = torch.from_numpy(np.asarray(uap).astype(dtype=np.float32))
                if len(uap.size()) == 4:
                    pass
                else:
                    uap = uap.unsqueeze(0)
                uap = F.interpolate(uap, size=(299, 299))
            else:
                TRAIN_TRANSFORMS = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ])
                TEST_TRANSFORMS = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor()
                ])
                if args.perturbation.endswith('.pth'):
                    uap = torch.load(args.perturbation)
                elif args.perturbation.endswith('.tar'):
                    ckpt = torch.load(args.perturbation)
                    try:
                        uap = ckpt['state_dict']['uap']
                    except:
                        uap = ckpt['uap']
                else:
                    uap = Image.open(args.perturbation).convert("RGB")
                    uap = np.asarray(uap) / 255
                    uap = np.transpose(uap, (2, 0, 1))
                    uap = torch.from_numpy(np.asarray(uap).astype(dtype=np.float32))

                if len(uap.size()) == 4:
                    pass
                else:
                    uap = uap.unsqueeze(0)
            args.model_name = model_name
            # dataset
            valdir = os.path.join(args.data_path, 'val')
            val_dataset = ImageFolder(root=valdir, transform=TEST_TRANSFORMS)
            acc, asr, fool_rate = evaluate_src_attack(args, val_dataset, uap, train_trans=TRAIN_TRANSFORMS, test_trans=TEST_TRANSFORMS)
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


def evaluate_downstream_dataset(args, uap_dict):
    dataset_names = ['birdsnap', 'caltech101', 'caltech256', 'dtd', 'fgvc', 'flowers', 'food', 'pet', 'stanford_cars', 'sun']
    if not isinstance(uap_dict, dict):
        uap_dict = [uap_dict]

    for key, value in uap_dict.items():
        args.perturbation = value
        curr_acc = []
        curr_asr = []
        curr_fool_rate = []
        for dataset in dataset_names:
            TEST_TRANSFORMS = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor()
            ])
            if args.perturbation.endswith('.pth'):
                uap = torch.load(args.perturbation)
            elif args.perturbation.endswith('.tar'):
                ckpt = torch.load(args.perturbation)
                try:
                    uap = ckpt['state_dict']['uap']
                except:
                    uap = ckpt['uap']
            else:
                uap = Image.open(args.perturbation).convert("RGB")
                uap = np.asarray(uap) / 255
                uap = np.transpose(uap, (2, 0, 1))
                uap = torch.from_numpy(np.asarray(uap).astype(dtype=np.float32))

            if len(uap.size()) == 4:
                pass
            else:
                uap = uap.unsqueeze(0)
            # dataset
            args.dataset = dataset
            data_path = eval(f"config.{dataset.upper()}_PATH")
            val_dataset = ImageNet(os.path.join(data_path, "test"), transform=TEST_TRANSFORMS)
            acc, asr, fool_rate = evaluate_attack_downstream_task(args, val_dataset, uap)
            curr_acc.append(acc)
            curr_asr.append(asr)
            curr_fool_rate.append(fool_rate)
        print(f"UAP method: {key}_{value}")
        print(f"UAP: {args.perturbation}")
        print("model", args.downstream_model_name)
        print('acc:', curr_acc)
        print('asr:', curr_asr)
        print('fool_rate:', curr_fool_rate)
        print("\n")

def check_files(uap_dict):
    for key, value in uap_dict.items():
        if not os.path.exists(value):
            sys.exit(f"file {value} not exit")

def check_model_ckpt(model_dict):
    for key, model_name in model_dict.items():
        model = pytorch_models[key](pretrained=True)
        

def load_downstream_model(args):
    ckpt_path = {
        'birdsnap':fr'{args.downstream_ckpt}\{args.finetune_type}\birdsnap\{args.model_name}_best_accuracy_model.pt',
        'caltech101':fr'{args.downstream_ckpt}\{args.finetune_type}\caltech101\{args.model_name}_best_accuracy_model.pt',
        'caltech256':fr'{args.downstream_ckpt}\{args.finetune_type}\caltech256\{args.model_name}_best_accuracy_model.pt',
        'dtd':fr'{args.downstream_ckpt}\{args.finetune_type}\dtd\{args.model_name}_best_accuracy_model.pt',
        'fgvc':fr'{args.downstream_ckpt}\{args.finetune_type}\fgvc_aircraft_2013b\{args.model_name}_best_accuracy_model.pt', # fgvc_aircraft_2013b
        'flowers':fr'{args.downstream_ckpt}\{args.finetune_type}\flowers\{args.model_name}_best_accuracy_model.pt',
        'food':fr'{args.downstream_ckpt}\{args.finetune_type}\food\{args.model_name}_best_accuracy_model.pt',
        'pet': fr'{args.downstream_ckpt}\{args.finetune_type}\pet\{args.model_name}_best_accuracy_model.pt',
        'cars':fr'{args.downstream_ckpt}\{args.finetune_type}\stanford_cars\{args.model_name}_best_accuracy_model.pt', # stanford_cars
        'sun':fr'{args.downstream_ckpt}\{args.finetune_type}\SUN397\{args.model_name}_best_accuracy_model.pt', # SUN397
    }
    return ckpt_path
    

if __name__ == "__main__":
    args = parser.parse_args()
    path = 'perturbations'

    pytorch_models = {
        'vgg16': models.vgg16,
        'vgg19': models.vgg19,
        'resnet50':models.resnet50,
        'resnet101':models.resnet101,
        'resnet152':models.resnet152,
        'resnext50': models.resnext50_32x4d,
        'wideresnet': models.wide_resnet50_2,
        'efficientnetb0': models.efficientnet_b0,
        'densenet121': models.densenet121,
        'densenet161': models.densenet161,
        'googlenet':models.googlenet,
        'mnasnet': models.mnasnet1_0,
        'alexnet': models.alexnet,
    }
    check_model_ckpt(pytorch_models)
    print("Model checked!!")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    TRAIN_TRANSFORMS = transforms.Compose([
        transforms.RandomResizedCrop(224) ,
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    TEST_TRANSFORMS = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224) ,
        transforms.ToTensor()
    ])

    IMGNET_MEAN = [0.485, 0.456, 0.406]
    IMGNET_STD = [0.229, 0.224, 0.225]
    normalize = Normalizer(mean=IMGNET_MEAN, std=IMGNET_STD)
    input_size = 224
    uap_files = {
        'our_uap_vgg16': f'{path}/our_uap_vgg16_eps10_fr9589.pth',
        'our_uap_vgg19': f'{path}/our_uap_vgg19_eps10_fr9478.pth',
        'our_uap_resnet50': f'{path}/our_uap_resnet50_eps10_fr9160.pth',
        'our_uap_resnet101': f'{path}/our_uap_resnet101_eps10_fr8774.pth',
        'our_uap_resnet152': f'{path}/our_uap_resnet152_eps10_fr8875.pth',
        'our_uap_resnext50': f'{path}/our_uap_resnext50_eps10_fr9255.pth',
        'our_uap_wideresnet50': f'{path}/our_uap_wideresnet_eps10_fr9134.pth',
        'our_uap_efficientnetb0': f'{path}/our_uap_efficientnetb0_eps10_fr8892.pth',
        'our_uap_densenet121': f'{path}/our_uap_densenet121_eps10_fr9177.pth',
        'our_uap_densenet161': f'{path}/our_uap_densenet161_eps10_fr9413.pth',
        'our_uap_alexnet': f'{path}/our_uap_alexnet_eps10_fr9327.pth',
        'our_uap_googlenet': f'{path}/our_uap_googlenet_eps10_fr9016.pth',
        'our_uap_mnasnet10': f'{path}/our_uap_mnasnet10_eps10_fr9576.pth',

    }
    check_files(uap_files)
    print("Check done!!！")

    if not args.is_downstream:
        evaluate_both_models(args, uap_files)
    else:
        evaluate_downstream_dataset(args, uap_files)
