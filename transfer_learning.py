import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import time

from data_loader import ImageNet
import config

parser = argparse.ArgumentParser(description="Settings: Train downstream model by using fixed or fullnet strategy")
parser.add_argument('--data_dir', type=str, default='', help="data path of training dataset")
parser.add_argument('--model_name', type=str, default='resnet50', help="the pretrained model used to finetuning")
parser.add_argument('--is_inception', type=bool, default=False, help="Whether the pretrained or downstream model is inception net")
parser.add_argument('--checkpoints', type=str, default='checkpoints/', help="the checkpoint path")
parser.add_argument('--num_classes', type=int, default=100, help="Category Number of downstream task")
parser.add_argument('--epochs', type=int, default=150, help="Max epoch to finetune downstream task")
parser.add_argument('--batch_size', type=int, default=64, help="Batch size per iteration")
parser.add_argument('--interval', type=int, default=10, help="Interval to print the training information")
parser.add_argument('--step_size', type=int, default=50, help="Training scheduler")
parser.add_argument('--gamma', type=int, default=0.1, help="Hyperparameter of the training scheduler")
parser.add_argument('--lr', type=float, default='1e-2', help="learning rate")
parser.add_argument('--weight_decay', type=float, default='5e-4', help="weight decay of the optimizer")
parser.add_argument('--momentum', type=float, default='0.9', help="momentum of the optimizer")


def process_model(model_name, model, num_classes, additional_hidden=0, mode='fullnet'):
    if mode == 'fixed':
        for param in model.parameters():
            param.requires_grad=False
    if model_name in ["resnet", "resnet18", "resnet50", "wide_resnet50_2", "wide_resnet50_4", "resnext50_32x4d", 'shufflenet','inception_v3']:
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


def freeze_model(model, freeze_level):
    update_params = []
    if freeze_level != -1:
        assert len([name for name, _ in list(model.named_parameters())
                    if f"layer{freeze_level}" in name]), "unknown freeze level (only {1,2,3,4} for ResNets)"
        update_params = []
        freeze = True
        for name, param in model.named_parameters():
            print(name, param.size())
            if not freeze and f'layer{freeze_level}' not in name:
                print(f"[Appending the params of {name} to the update list]")
                update_params.append(param)
            else:
                param.requires_grad = False

            if freeze and f'layer{freeze_level}' in name:
                # if the freeze level is detected stop freezing onwards
                freeze = False
    return update_params


def get_update_params(model, model_name):
    if 'vgg' in model_name:
        updata_params = model.classifier[6].parameters()
    elif 'densenet' in model_name:
        updata_params = model.classifier.parameters()
    else:
        updata_params = model.fc.parameters()
    return updata_params


def fixed_feature_transfer_learning(args):
    if 'train' in os.listdir(args.data_dir):
        train_dataset = ImageNet(os.path.join(args.data_dir, "train"), transform=TRAIN_TRANSFORMS)
        val_dataset = ImageNet(os.path.join(args.data_dir, "test"), transform=TEST_TRANSFORMS)
    else:
        dataset = torchvision.datasets.ImageNet(args.data_dir, transform=TRAIN_TRANSFORMS)
        training_samples = int(dataset.__len__() * 0.8)
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, lengths=[training_samples,
                                                                                     dataset.__len__() - training_samples],
                                                                   generator=torch.Generator().manual_seed(1024))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=12, pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=True)

    model = pytorch_models[args.model_name](pretrained=True)
    model, input_size = process_model(args.model_name, model, train_dataset.num_classes, mode='fixed')
    model = model.to(device)
    updata_params = get_update_params(model, args.model_name)
    optimizer = torch.optim.SGD(updata_params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    schuler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_size, gamma=args.gamma)

    best_loss = np.inf
    best_acc = 0
    correct = 0
    n = 0
    for epoch in range(args.epochs):
        phar = tqdm(train_loader, desc="train loader")
        model.train()
        for i, (images, labels) in enumerate(phar):
            images = images.to(device)
            labels = labels.to(device)
            if args.is_inception:
                output, aux_outputs = model(images)
                loss1 = criterion(output, labels)
                loss2 = criterion(aux_outputs, labels)
                loss = loss1 + 0.4 * loss2
            else:
                output = model(images)
                loss = criterion(output, labels)

            model.zero_grad()
            loss.backward()
            optimizer.step()
            pred = torch.argmax(output, dim=1)
            correct += (labels == pred).sum()
            n += images.size(0)
            phar.set_description(
                "【{}/{}】: loss: {}, accuracy:{}".format(epoch, args.epochs, loss.item(), 100 * (correct / n)))

        if (epoch + 1) % args.interval == 0:
            model.eval()
            val_correct = 0
            total = 0
            for i, (images, labels) in enumerate(val_loader):
                images = images.to(device)
                labels = labels.to(device)

                loss = criterion(output, labels)
                output = model(images)
                pred = torch.argmax(output, dim=1)
                val_correct += (labels == pred).sum()
                total += images.size(0)
            val_acc = val_correct / total
            print("validation accuracy:{:.2f}".format(100 * val_acc))
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), f"{log_dir}/best_accuracy_model.pt")
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(model.state_dict(), f"{log_dir}/best_loss_model.pt")
            torch.save(model.state_dict(), f"{log_dir}/last_model.pt")
        schuler.step(epoch)


def full_network_transfer_learning(args):
    if 'train' in os.listdir(args.data_dir):
        train_dataset = ImageNet(os.path.join(args.data_dir, "train"), transform=TRAIN_TRANSFORMS)
        val_dataset = ImageNet(os.path.join(args.data_dir, "test"), transform=TEST_TRANSFORMS)
    else:
        dataset = torchvision.datasets.ImageNet(args.data_dir, transform=TRAIN_TRANSFORMS)
        training_samples = int(dataset.__len__() * 0.8)
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, lengths=[training_samples,
                                                                                     dataset.__len__() - training_samples],
                                                                   generator=torch.Generator().manual_seed(1024))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=12, pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=True)

    model = pytorch_models[args.model_name](pretrained=True)
    model, input_size = process_model(args.model_name, model, train_dataset.num_classes)
    model = model.to(device)
    if 'vgg' in args.model_name:
        ignored_params = list(map(id, model.classifier[6].parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    elif 'densenet' in args.model_name:
        ignored_params = list(map(id, model.classifier.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    else:
        ignored_params = list(map(id, model.fc.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    updata_params = get_update_params(model, args.model_name)
    optimizer = torch.optim.SGD(
        [{'params': base_params},
         {'params': updata_params, 'lr': args.lr}], lr=args.lr * 0.1, momentum=args.momentum,
        weight_decay=args.weight_decay)
    schuler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_size, gamma=args.gamma)

    best_loss = np.inf
    best_acc = 0
    correct = 0
    n = 0
    for epoch in range(args.epochs):
        phar = tqdm(train_loader, desc="train loader")
        model.train()
        for i, (images, labels) in enumerate(phar):
            images = images.to(device)
            labels = labels.to(device)
            if args.is_inception:
                output, aux_outputs = model(images)
                loss1 = criterion(output, labels)
                loss2 = criterion(aux_outputs, labels)
                loss = loss1 + 0.4 * loss2
            else:
                output = model(images)
                loss = criterion(output, labels)

            model.zero_grad()
            loss.backward()
            optimizer.step()
            pred = torch.argmax(output, dim=1)
            correct += (labels == pred).sum()
            n += images.size(0)
            phar.set_description(
                "【{}/{}】: loss: {}, accuracy:{}".format(epoch, args.epochs, loss.item(), 100 * (correct / n)))
        with torch.no_grad():
            if (epoch + 1) % args.interval == 0:
                model.eval()
                val_correct = 0
                total = 0
                for i, (images, labels) in enumerate(val_loader):
                    images = images.to(device)
                    labels = labels.to(device)

                    loss = criterion(output, labels)
                    output = model(images)
                    pred = torch.argmax(output, dim=1)
                    val_correct += (labels == pred).sum()
                    total += images.size(0)
                val_acc = val_correct / total
                print("validation accuracy:{:.2f}".format(100 * val_acc))
                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save(model.state_dict(), f"{log_dir}/best_accuracy_model.pt")
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    torch.save(model.state_dict(), f"{log_dir}/best_loss_model.pt")
                torch.save(model.state_dict(), f"{log_dir}/last_model.pt")
        schuler.step(epoch)


def train_network_from_scratch(args):
    if 'train' in os.listdir(args.data_dir):
        train_dataset = ImageNet(os.path.join(args.data_dir, "train"), transform=TRAIN_TRANSFORMS)
        val_dataset = ImageNet(os.path.join(args.data_dir, "test"), transform=TEST_TRANSFORMS)
    else:
        dataset = torchvision.datasets.ImageNet(args.data_dir, transform=TRAIN_TRANSFORMS)
        training_samples = int(dataset.__len__() * 0.8)
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, lengths=[training_samples,
                                                                                     dataset.__len__() - training_samples],
                                                                   generator=torch.Generator().manual_seed(1024))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=12, pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=True)

    model = pytorch_models[args.model_name](pretrained=False)
    model, input_size = process_model(args.model_name, model, train_dataset.num_classes)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
        weight_decay=args.weight_decay)
    schuler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_size, gamma=args.gamma)

    best_loss = np.inf
    best_acc = 0
    correct = 0
    n = 0
    for epoch in range(args.epochs):
        phar = tqdm(train_loader, desc="train loader")
        model.train()
        for i, (images, labels) in enumerate(phar):
            images = images.to(device)
            labels = labels.to(device)
            if args.is_inception:
                output, aux_outputs = model(images)
                loss1 = criterion(output, labels)
                loss2 = criterion(aux_outputs, labels)
                loss = loss1 + 0.4 * loss2
            else:
                output = model(images)
                loss = criterion(output, labels)

            model.zero_grad()
            loss.backward()
            optimizer.step()
            pred = torch.argmax(output, dim=1)
            correct += (labels == pred).sum()
            n += images.size(0)
            phar.set_description(
                "【{}/{}】: loss: {}, accuracy:{}".format(epoch, args.epochs, loss.item(), 100 * (correct / n)))
        with torch.no_grad():
            if (epoch + 1) % args.interval == 0:
                model.eval()
                val_correct = 0
                total = 0
                for i, (images, labels) in enumerate(val_loader):
                    images = images.to(device)
                    labels = labels.to(device)

                    loss = criterion(output, labels)
                    output = model(images)
                    pred = torch.argmax(output, dim=1)
                    val_correct += (labels == pred).sum()
                    total += images.size(0)
                val_acc = val_correct / total
                print("validation accuracy:{:.2f}".format(100 * val_acc))
                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save(model.state_dict(), f"{log_dir}/best_accuracy_model.pt")
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    torch.save(model.state_dict(), f"{log_dir}/best_loss_model.pt")
                torch.save(model.state_dict(), f"{log_dir}/last_model.pt")
        schuler.step(epoch)


def make_log_dir(exp_dict):
    dir_name = ''
    for key, value in exp_dict.items():
        dir_name += '{}_{}_'.format(key, value)
    dir_name = 'logs/' + dir_name
    print(dir_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name

if __name__ == "__main__":
    args = parser.parse_args()
    cfg = config.DTD_PATH
    args.data_dir = cfg['path']
    folder_name = {
        'mode': 'train_from_scratch',
        'dataset': "DTD",
        'date': time.strftime("%m-%d-%H-%M", time.localtime()),
        'model_name': args.model_name
    }
    log_dir = make_log_dir(folder_name)
    pytorch_models = {
        'alexnet': models.alexnet,
        'vgg16': models.vgg16,
        'vgg16_bn': models.vgg16_bn,
        'squeezenet': models.squeezenet1_0,
        'densenet': models.densenet161,
        'shufflenet': models.shufflenet_v2_x1_0,
        'mobilenet': models.mobilenet_v2,
        'resnext50_32x4d': models.resnext50_32x4d,
        'resnet50': models.resnet50,
        'mnasnet': models.mnasnet1_0,
        'inception_v3': models.inception_v3,
    }

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    criterion = nn.CrossEntropyLoss()

    TRAIN_TRANSFORMS = transforms.Compose([
        transforms.RandomResizedCrop(224),
        # transforms.RandomResizedCrop(299), # used for inception net
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    TEST_TRANSFORMS = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        # transforms.Resize(330),  # used for inception net
        # transforms.CenterCrop(299), # used for inception net
        transforms.ToTensor()
    ])
    fixed_feature_transfer_learning(args)
    full_network_transfer_learning(args)
    train_network_from_scratch(args)
