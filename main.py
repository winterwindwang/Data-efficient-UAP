import numpy as np
import torch
import  time
import random
import torchvision
import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import argparse
import os
from uap_utils.utils import get_model
from uap_utils.feature_extractor import FeatureExtractor
from data_loader import get_dataset
import pathlib
import torch.nn.functional as F


def get_args():
    parser = argparse.ArgumentParser(description='Parameters loader')
    parser.add_argument('--exp_name', type=str, default=None, help='If not specific, the exp_name is assigned as [model_name]_[train_data_name]_[epsilon]_[time]')
    parser.add_argument('--model_name', type=str, default='resnet50', help='The victim model to attack')
    parser.add_argument('--train_data_name', choices=['imagenet', 'coco', 'voc', 'sun397', 'mixed'], default='imagenet',
                        help='Choice the dataset to train the UAP, the default is imagenet. Note that mixed should be manner assign in get_dataset as it contains four path')
    parser.add_argument('--test_data_name', type=str, default='imagenet', help='The dataset used to test the UAP, available: ')
    parser.add_argument('--train_data_path', type=str, default='', help='The training dataset dir')
    parser.add_argument('--test_data_path', type=str, default='', help='The test dataset dir')
    parser.add_argument('--save_dir', type=str, default='checkpoints/', help='The dir path to save the uap')
    parser.add_argument('--eps', type=float, default=10., help='epsilon, limit the perturbation to [-10, 10] respect to [0, 255]')
    parser.add_argument('--input_size', type=int, default=224, help='The image size')
    parser.add_argument('--batch_size', type=int, default=50, help='The batch size per ieteration')
    parser.add_argument('--epochs', type=int, default=50, help='The max epoch to train the UAP')
    parser.add_argument('--miu', type=float, default=0.1, help="The decay factor of the momentum")
    parser.add_argument('--eps_step', type=float, default=0.001, help="The update step, which depends on the networks")
    parser.add_argument('--num_works', type=int, default=8, help='Number workers of the Dataloader')
    parser.add_argument('--nb_images', type=int, default=2000, help='Number of image used to train the UAP')
    parser.add_argument('--feat_type', choices=['half_feat_and_grad', 'all_feat_and_grad', 'all_feat', 'half_feat'], default="half_feat_and_grad",
                        help="Choice the feature arrange manner from ['half_feat_and_grad', 'all_feat_and_grad', 'all_feat', 'half_feat'](default is half_feat_and_grad)")
    parser.add_argument('--loss_type', choices=['abs', 'square'], default="abs",
                        help="Choice the loss type from ['abs', 'square'] (default is the abs)")
    parser.add_argument('--sort_type', choices=['mae', 'cos_similarity', 'channel_mean', 'nonzero', 'gradient', 'random'], default="channel_mean",
                        help="Choice the feature sort type from ['mae', 'cos_similarity', 'channel_mean', 'nonzero', 'gradient', 'random'](default is channel_mean)")
    args = parser.parse_args()
    return args

def seed_torch(seed=1024):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    np.random.seed(1024)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.

@torch.no_grad()
def evaluate_pert(model, val_loader, uap):
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
    print("Total:{}, success pred: {}, success attack: {}, fool number: {}".format(n, ori_correct, correct, fool_num))
    return np.round(100 * (ori_correct / n), 2), np.round(100 * (correct / n), 2), np.round(
        100 * (fool_num / n), 2)


def train(iter_idx=0):
    extractor = FeatureExtractor(model, args.model_name)

    batch_delta = torch.autograd.Variable(torch.zeros([args.batch_size, 3, args.input_size, args.input_size]),
                                          requires_grad=True)  # initialize as zero vector
    delta = batch_delta[0]

    layer_str = f'{args.model_name}_{args.sort_type}_sorted_{args.train_data_name}_img{args.nb_images}_{args.feat_type}_{args.loss_type}_epoch{args.epochs}_eps{args.eps}_{iter_idx}'
    save_path = pathlib.Path(
        f"saved_images/{args.exp_name}/{layer_str}")
    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)
    print(save_path)
    batch_delta.requires_grad_()

    for epoch in range(args.epochs):
        momentum = torch.zeros_like(delta).detach()
        phar = tqdm.tqdm(train_loader, disable=True)
        for images, labels in phar:
            batch_delta.data = delta.unsqueeze(0).repeat([images.shape[0], 1, 1, 1])
            adv_images = torch.clamp((images + batch_delta).to(device), 0, 1)
            model_output, loss = extractor.run(adv_images, sort_type=args.sort_type, feat_type=args.feat_type, loss_type=args.loss_type)
            loss.backward()

            # momentum
            grad = batch_delta.grad.data
            grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
            grad = grad + args.miu * momentum
            momentum = grad
            grad = grad.mean(dim=0)

            delta = delta + args.eps_step * grad.sign()
            delta = torch.clamp(delta, -args.eps/255, args.eps/255)

            batch_delta.grad.data.zero_()

        print(f"Epoch: {epoch}/{args.epochs}, Loss: {loss.item()}, uap mean: {delta.mean().item()}")
        if args.epochs >=100:
            if (epoch+1) % 20 == 0:
                transforms.ToPILImage()(delta).save(save_path / f"sauap_{epoch}.png")
        else:
            if (epoch+1) % 5 == 0:
                transforms.ToPILImage()(delta).save(save_path / f"sauap_{epoch}.png")
    if args.model_name == 'inception_v3':
        delta = F.interpolate(delta.unsqueeze(0), size=(224, 224))
        delta = delta[0]
    extractor.clear_hook()
    
    uap_save_path = pathlib.Path(
        f"{args.save_dir}/{args.exp_name}")
    if not uap_save_path.exists():
        uap_save_path.mkdir(parents=True, exist_ok=True)
    torch.save(delta.data, uap_save_path / f"{layer_str}.pth")
    print(uap_save_path / f"{layer_str}.pth")
    return delta.data


# for different models
if __name__ == "__main__":
    seed_torch(1024)
    device = 'cuda:0' if torch.cuda.is_available() else 'cuda'
    args = get_args()
    model, args = get_model(args)
    model.to(device)
    model.eval()

    if args.exp_name is None:
        time_str = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
        args.exp_name = f"{args.model_name}_{args.train_data_name}_{args.eps}_{time_str}"

    train_data = get_dataset(data_name=args.train_data_name, data_path=args.train_data_path, nb_images=args.nb_images)
    test_data = get_dataset(data_name=args.test_data_name, data_path=args.test_data_path, only_train=False)

    sample_indices = np.random.permutation(range(train_data.__len__()))[:args.nb_images]
    train_data = Subset(train_data, sample_indices)

    five_run_hist_fr = []
    five_run_hist_acc = []
    five_run_hist_asr = []
    for iter_epoch in range(5):
        if "densenet161" in args.model_name:
            args.batch_size = 25
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                                pin_memory=True, num_workers=args.num_works)
        validation_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size,
                                                    num_workers=args.num_works)
        uap = train(iter_epoch)

        # evaluate
        acc, asr, fooling_rate = evaluate_pert(model, validation_loader, uap.to(device))
        five_run_hist_fr.append(fooling_rate)
        five_run_hist_acc.append(acc)
        five_run_hist_asr.append(asr)
    print(f"model: {args.model_name}")
    print(f"validtion: acc: {five_run_hist_acc}, fooling rate:{five_run_hist_fr}, asr: {five_run_hist_asr}")
    print(f"mean value: acc:{np.mean(five_run_hist_acc)},  fooling rate:{np.mean(five_run_hist_fr)}, asr:{np.mean(five_run_hist_asr)}")