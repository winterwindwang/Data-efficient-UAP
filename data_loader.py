import json
import os
import shutil
from typing import List, Any
import torchvision.models
from torch.utils.data import Dataset, Subset
from PIL import Image
import torchvision.datasets as dset
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
import torch
from glob import glob
import random


def default_fn(file):
    img = Image.open(file).convert("RGB")
    return img

class ImageNet(Dataset):
    def __init__(self, data_folder, label_path='', transform=None, default_fn=default_fn):
        data = []
        classes = {}
        if label_path:
            with open(label_path, 'r') as fr:
                lines = fr.readlines()
                # lines = fr.read()
                labels = []
                for line in lines:
                    img_name, label = line.split()
                    data.append((os.path.join(data_folder, img_name), int(label)))
                    labels.append(int(label))
                # num_classes=len(set(labels))
                for i in range(len(set(labels))):
                    classes[i] = i
        else:
            data_list = []
            for pth in data_folder:
                if "COCO" in pth or 'VOC' in pth:
                    file_list = glob(os.path.join(pth, "*.jpg"))
                    random.shuffle(file_list)
                    file_list = file_list[:5000]
                    data_list.extend(file_list)
                else:
                    current_list = []
                    for subfolder in os.listdir(pth):
                        if "SUN397" in pth:
                            file_list = glob(os.path.join(pth, subfolder, "*.jpg"))
                            current_list.extend(file_list)
                        else:
                            file_list = glob(os.path.join(pth, subfolder, "*.JPEG"))
                            current_list.extend(file_list)
                    random.shuffle(current_list)
                    current_list = current_list[:5000]
                    data_list.extend(current_list)
            data = data_list
        self.data = data
        self.label_path = label_path
        self.classes = classes
        self.num_classes = len(classes.values())
        self.transform = transform
        self.default_fn = default_fn

    def __len__(self):
        return len(self.data)

    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, index):
        if self.label_path:
            file, label = self.data[index]
        else:
            file = self.data[index]
            label = 1
        img = self.default_fn(file)

        if self.transform is not None:
            img = self.transform(img)

        return img, label

class COCOImage(Dataset):
    def __init__(self, data_folder, transform=None, default_fn=default_fn):
        data = []
        classes = {}

        dirlists = os.listdir(data_folder)
        for file in dirlists:
            filepath = os.path.join(data_folder, file)
            data.append((filepath, 1))
        self.data = data
        self.classes = 1000
        self.num_classes = len(classes.values())
        self.transform = transform
        self.default_fn = default_fn

    def __len__(self):
        return len(self.data)

    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, index):
        file, label = self.data[index]

        img = self.default_fn(file)
        
        if self.transform is not None:
            img = self.transform(img)

        return img, label

class SubTaskDataset(Dataset):
    def __init__(self, data_folder, label_path, transform=None, default_fn=default_fn):
        data = []
        with open(label_path, 'r') as fr:
            lines = fr.readlines()
            # lines = fr.read()
            for line in lines:
                img_name, label = line.split()
                data.append((os.path.join(data_folder, img_name), int(label)))
        self.data = data
        self.transform = transform
        self.default_fn = default_fn

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        file, label = self.data[index]

        img = self.default_fn(file)

        if self.transform is not None:
            img = self.transform(img)

        return img, label

class ImageNetSplit(Dataset):
    def __init__(self, data_folder, label_path='', transform=None, default_fn=default_fn):
        data = []
        classes = {}
        if label_path:
            with open(label_path, 'r') as fr:
                lines = fr.readlines()
                # lines = fr.read()
                labels = []
                for line in lines:
                    img_name, label = line.split()
                    data.append((os.path.join(data_folder, img_name), int(label)))
                    labels.append(int(label))
                # num_classes=len(set(labels))
                for i in range(len(set(labels))):
                    classes[i] = i
        else:
            dirlists = os.listdir(data_folder)
            for i, dirfile in enumerate(dirlists):
                subdirname = os.path.join(data_folder, dirfile)
                classes[dirfile] = i
                for file in os.listdir(subdirname):
                    filepath = os.path.join(subdirname, file)
                    data.append((filepath, i))
        self.data = data
        self.classes = classes
        self.num_classes = len(classes.values())
        self.transform = transform
        self.default_fn = default_fn

    def __len__(self):
        return len(self.data)

    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, index):
        file, label = self.data[index]

        img = self.default_fn(file)

        if self.transform is not None:
            img = self.transform(img)
        return img, label, file


from pycocotools.coco import COCO
class CocoDataset(Dataset):
    def __init__(self, root, annFile, target_transform=None, transform=None):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = root
        self.coco = COCO(annFile)
        ids = list(self.coco.imgs.keys())
        ids_no_anno = [3799,9759,10263,16689, 16903, 13789, 10440,13466,21382,23017,24499,25378,26767,28156,29056,29594,33123,33422,33554,34089,39068,43947,45075,46633,48546,49255,49883,50637,52726,58133,60434,62805,62824,66543,68715,68838,69911,70125,72912,75083,75426,75481,78947,79331,79362,79913,83246,84018,88517,90280,91372,92554,92604,95772,96923,98268,101535,101623,104829,104880,108169,109942,113185,118615,120235,123239,124983,125084,125182,125997,127626,129903,129988,130712,133693,135042,136173,140974,150616,150779,152732,156299,156606,162539,166260,166524,173685,174902,176149,176168,176193,176649,179430,181462,184874,189740,191501,192062,192817,193704,197774,204435,208708,209420,209630,210766,211423,211665,212675,217005,218357,220160,220739,221828,222330,222757,224742,224970,226128,226629,228415,229782,229981,230795,237860,238141,240436,241209,244160,244636,246348,246973,247624,252101,253520,253688,262189,262284,266091,266611,268496,268693,268770,273123,274233,274957,276731,277329,279103,280413,281188,281582,283147,285068,289943,293671,294370,294431,297736,298190,299045,300090,301765,304036,305159,305871,306477,308828,309222,309571,311877,315846,317120,317130,317575,318596,321603,322887,325357,325368,325690,328084,328098,330535,331600,333198,334642,335669,336777,336873,337506,337653,338067,339740,340781,342335,344730,345063,345711,349579,350334,354041,357948,358795,359184,359276,359465,361831,362696,362986,368884,375363,379190,381842,382115,382191,382333,383450,386613,389811,392534,393212,395124,395185,397089,397278,397287,398454,401212,401623,402869,403104,403279,404462,404871,405104,405815,405945,406217,407976,413120,415714,419106,421673,421970,425933,427094,428399,429386,431026,431234,432373,432647,433971,434129,441788,441863,444302,449546,450098,450343,451373,459408,459590,464296,466935,466958,468935,474398,486632,487702,489907,491482,493956,495053,498975,500079,500780,503200,503483,503860,507686,509423,513149,516542,519359,521098,522127,529578,531232,533896,534918,539390,540388,544597,545006,545235,547047,547760,547903,549012,551550,553420,559576,562582,564317,566025,566103,568863,570045,571242,572546,573113,576017,576354,577949,578852,579247,581087, 25593, 41488, 42888, 49091, 58636, 64574, 98497, 101022, 121153, 127135, 173183, 176701, 198915, 200152, 226111, 228771, 240767, 260657, 261796, 267946, 268996, 270386, 278006, 308391, 310622, 312549, 320706, 330554, 344611, 370999, 374727, 382734, 402096, 404601, 447789, 458790, 461275, 476491, 477118, 481404, 502910, 514540, 528977, 536343, 542073, 550939, 556498, 560371]
        self.ids = list(sorted(list(set(ids) - set(ids_no_anno))))

        cats = self.coco.loadCats(self.coco.getCatIds())
        cat_nms = [cat['name'] for cat in cats]
        self.num_class = len(cat_nms) + 1
        self.transform = target_transform
        self.transforms = transform

    def __len__(self):
        return len(self.ids)

    def _load_target(self, id) -> List[Any]:
        anns = self.coco.loadAnns(self.coco.getAnnIds(id))
        mask = torch.LongTensor(np.max(np.stack([self.coco.annToMask(ann) * ann["category_id"]
                                                 for ann in anns]), axis=0)).unsqueeze(0)
        return mask

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")


    def __getitem__(self, index):
        """Return one data pair (image and mask)"""
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)
        if self.transform is not None:
            image, target = self.transform(image, target)
        if self.transforms is not None:
            image = self.transforms(image)
        path = os.path.join(self.root, self.coco.loadImgs(id)[0]["file_name"])
        return image, target, path



class COCODataset(Dataset):
    def __init__(self, data_folder, transform=None, default_fn=default_fn):
        data = []
        for file in os.listdir(data_folder):
            data.append((file, 1))
        self.data = data
        self.transform = transform
        self.default_fn = default_fn

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        file, label = self.data[index]
        img = self.default_fn(file)
        if self.transform is not None:
            img = self.transform(img)
        return img, label, file


class COCOVOCDataset(Dataset):
    def __init__(self, data_folder, transform=None, default_fn=default_fn):
        data = []
        for file in os.listdir(data_folder):
            file_path = os.path.join(data_folder, file)
            data.append((file_path, 1))
        self.data = data
        self.transform = transform
        self.default_fn = default_fn

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        file, label = self.data[index]
        img = self.default_fn(file)
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def get_dataset(data_name, data_path, nb_images=500, input_size=224, only_train=True):
    if data_name == 'imagenet':
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(input_size),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
        ])
        traindir = os.path.join(data_path, 'ImageNet10k')
        valdir = os.path.join(data_path, 'val')
        # dataset
        if only_train:
            train_dataset = ImageFolder(root=traindir, transform=train_transform)
            if nb_images < 10000:
                random.seed(1024)
                np.random.seed(1024)
                sample_indices = np.random.permutation(range(train_dataset.__len__()))[:nb_images]
                print(sample_indices)
                train_dataset = Subset(train_dataset, sample_indices)
        else:
            train_dataset = ImageFolder(root=valdir, transform=test_transform)
    elif data_name == 'coco':
        # COCO2014/train2014/
        train_transform = transforms.Compose([
            transforms.Resize(int(input_size * 1.143)),
            transforms.RandomCrop(input_size),
            transforms.ToTensor(),
        ])
        train_dataset = COCOVOCDataset(data_path, train_transform)
        if nb_images < 50000:
            random.seed(1024)
            np.random.seed(1024)
            sample_indices = np.random.permutation(range(train_dataset.__len__()))[:nb_images]
            train_dataset = Subset(train_dataset, sample_indices)
    elif data_name == 'voc':
        # path = 'dataset/VOC200712/VOCdevkit/VOC2012/JPEGImages/'
        train_transform = transforms.Compose([
            transforms.Resize(int(input_size * 1.143)),
            transforms.RandomCrop(input_size),
            transforms.ToTensor(),
        ])
        train_dataset = COCOVOCDataset(data_path, train_transform)
        if nb_images < 50000:
            random.seed(1024)
            np.random.seed(1024)
            sample_indices = np.random.permutation(range(train_dataset.__len__()))[:nb_images]
            train_dataset = Subset(train_dataset, sample_indices)
    elif data_name == 'sun397':
        path = 'dataset/transfer/SUN397/'
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(input_size),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
        ])
        traindir = os.path.join(data_path, 'train')
        valdir = os.path.join(data_path, 'test')
        # dataset
        if only_train:
            train_dataset = ImageFolder(root=traindir, transform=train_transform)
            if nb_images < 50000:
                random.seed(1024)
                np.random.seed(1024)
                sample_indices = np.random.permutation(range(train_dataset.__len__()))[:nb_images]
                train_dataset = Subset(train_dataset, sample_indices)
        else:
            train_dataset = ImageFolder(root=valdir, transform=test_transform)
    elif data_name == 'mixed':
        # data_path = [
        #     'dataset/transfer/SUN397/',
        #     'dataset/COCO/train2014/',
        #     'dataset/ImageNet/ImageNet10k/',
        #     'dataset/VOC200712/VOCdevkit/VOC2012/JPEGImages/'
        # ]
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(input_size),
            transforms.ToTensor(),
        ])
        # dataset
        if only_train:
            train_dataset = ImageNet(data_folder=data_path, transform=train_transform)
            if nb_images < 50000:
                random.seed(1024)
                np.random.seed(1024)
                sample_indices = np.random.permutation(range(train_dataset.__len__()))[:nb_images]
                train_dataset = Subset(train_dataset, sample_indices)
        else:
            pass
    return train_dataset


if __name__ == "__main__":
    from torchvision import transforms
    from torch.utils.data import DataLoader
    import torch

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    data_folder = r'F:\DataSource\ImageNet\ILSVRC2012_img_val'
    label_path = r'F:\DataSource\ImageNet\val.txt'

    # dataset = ImageNetSplit(data_folder, label_path, transform)
    # # dataset = ImageNet(r'F:\DataSource\AttackTransfer\caltech101\101_ObjectCategories', transform=transform)
    #
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=False, pin_memory=True)  # , num_workers=4
    model = torchvision.models.resnet50(pretrained=True)
    model = model.cuda()
    model.eval()


    # ImageFolder
    input_size = 224
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    traindir = r'D:\DataSource\ImageNet\ImageNet-10k'
    valdir = "D:/DataSource/ImageNet/val/"
    label_json = r'D:\DataSource\ImageNet\imagenet_class_index.json'
    # with open(label_json, 'r') as f:
    #     labels_json = json.load(f)
    # lab2idx = {}
    # name2label = {}
    # for key, value in labels_json.items():
    #     lab2idx[value[1]] = int(key)
    #     name2label[value[1].lower()] = value[0]
    #
    # def rename(path):
    #     filenames = os.listdir(path)
    #     for filename in filenames:
    #         lowerfilename = name2label[filename.lower()]
    #         file = os.path.join(path, filename)
    #         new_file = os.path.join(path, lowerfilename)
    #         # if os.path.isdir(file):
    #         #     rename(file)
    #         # else:
    #         os.rename(file, new_file)
    #
    #
    # rename(valdir)

    train_transform = transforms.Compose([
        transforms.Resize(256),
        # transforms.Resize(299), # inception_v3
        transforms.RandomCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        # transforms.Resize(299), # inception_v3
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])

    train_data = dset.ImageFolder(root=traindir, transform=train_transform)
    test_data = dset.ImageFolder(root=valdir, transform=test_transform)

    dataloader = DataLoader(test_data, batch_size=32, shuffle=False, pin_memory=True)  # , num_workers=4

    save_dir = r'F:\DataSource\AttackTransfer\ImageNet'
    correct = 0
    count = [0] * 1000
    for i, (images, labels) in enumerate(dataloader):
        images = images.cuda()
        labels = labels.cuda()

        output = model(images)
        preds = torch.argmax(output,dim=1)
        correct += (preds == labels).sum()
        # for pred, label, file in zip(preds, labels, files):
        #     if pred == label:
        #         dst_dir = os.path.join(save_dir, str(label.item()))
        #         if not os.path.exists(dst_dir):
        #             os.makedirs(dst_dir)
        #         filename = os.path.basename(file)
        #         src = file
        #         dst = os.path.join(dst_dir, filename)
        #         shutil.copy(src, dst)
        #         count[label] += 1



    print(correct)
    print(test_data.__len__())
    acc = correct / test_data.__len__()
    print("Accuracy is ", (acc))  ## 0.7559,  官网给出的： 76.130

    # tensor(9699, device='cuda:0')
    # 10000
    # Accuracy is tensor(0.9699, device='cuda:0')

    # tensor(38074, device='cuda:0')
    # 50000
    # Accuracy is tensor(0.7615, device='cuda:0')