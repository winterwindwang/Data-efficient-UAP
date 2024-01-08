import torch
import torchvision
import matplotlib.pyplot as plt
from torch.nn import functional as F
import random
import numpy as np


class FeatureExtractor(object):
    # features = None
    gradient = None

    def __init__(self, model, model_name, type='gduap'):
        self.model = model
        self.features_handles = []

        if 'vgg' in model_name:
            self.features_handles.append(self.model[1].features[2].register_forward_hook(self._hook_fn_gduap))
            self.features_handles.append(self.model[1].features[2].register_backward_hook(self._hook_gn))
        elif 'resnet' in model_name:
            self.features_handles.append(self.model[1]._modules.get("relu").register_forward_hook(self._hook_fn_gduap))
            self.features_handles.append(self.model[1]._modules.get("relu").register_backward_hook(self._hook_gn))
        elif "densenet" in model_name:
            self.features_handles.append(self.model[1].features.relu0.register_forward_hook(self._hook_fn_gduap))
            self.features_handles.append(self.model[1].features.relu0.register_backward_hook(self._hook_gn))
        elif "googlenet" in model_name:
            self.features_handles.append(self.model[1].inception4a.register_forward_hook(self._hook_fn_gduap))
            self.features_handles.append(self.model[1].inception4a.register_backward_hook(self._hook_gn))
        elif "alexnet" in model_name:
            self.features_handles.append(self.model[1].features[0].register_forward_hook(self._hook_fn_gduap))
            self.features_handles.append(self.model[1].features[0].register_backward_hook(self._hook_gn))
        elif "mnasnet10" in model_name:
            self.features_handles.append(self.model[1].layers[0].register_forward_hook(self._hook_fn_gduap))
            self.features_handles.append(self.model[1].layers[0].register_backward_hook(self._hook_gn))
        elif "resnext50" in model_name:
            self.features_handles.append(self.model[1].conv1.register_forward_hook(self._hook_fn_gduap))
            self.features_handles.append(self.model[1].conv1.register_backward_hook(self._hook_gn))
        elif "wideresnet" in model_name:
            self.features_handles.append(self.model[1].conv1.register_forward_hook(self._hook_fn_gduap))
            self.features_handles.append(self.model[1].conv1.register_backward_hook(self._hook_gn))
        elif "efficientnetb0" in model_name:
            self.features_handles.append(self.model[1].features[0].register_forward_hook(self._hook_fn_gduap))
            self.features_handles.append(self.model[1].features[0].register_backward_hook(self._hook_gn))
        elif "inception_v3" in model_name:
            self.features_handles.append(self.model[1].Conv2d_1a_3x3.register_forward_hook(self._hook_fn_gduap))
            self.features_handles.append(self.model[1].Conv2d_1a_3x3.register_backward_hook(self._hook_gn))

    def _hook_fn_gduap(self, module, input_feature, output_feature):
        self.loss_feature.append(output_feature)

    def _hook_fn_fff(self, module, input_feature, output_feature):
        loss = torch.log((torch.mean(torch.abs(output_feature))))
        self.loss_feature += loss

    def _hook_fn_dat(self, module, input_feature, output_feature):
        loss = torch.log((torch.mean(torch.abs(output_feature))))
        self.loss_feature += loss

    def _hook_gn(self, module, grad_in, grad_out):
        self.gradient = grad_out

    def _back_prop(self, score, class_idx):
        loss = score[:, class_idx].sum()
        self.model.zero_grad()
        loss.backward(retain_graph=True)

    def _get_weights(self, score, class_idx):
        self._back_prop(score, class_idx)
        # return self.gradient.squeeze(0).mean(axis=(1, 2))
        return self.gradient

    def clear_hook(self):
        for handle in self.features_handles:
            handle.remove()

    def run(self, _input, _class_idx=None, sort_type=None, **kwargs):
        self.loss_feature = []
        loss = torch.tensor(0.)
        self.model.zero_grad()
        output = self.model(_input)

        _class_idx_max = torch.argmax(output, dim=1)
        weights_max = self._get_weights(output, _class_idx_max)[0].detach()
        weight_shape = weights_max.size()
        _input = torch.index_select(_input, dim=1, index=torch.tensor([0]).cuda())


        if sort_type == 'mae':
            _input = F.interpolate(_input, self.loss_feature[0].size()[-2:]).detach()
            mae = torch.mean(_input - self.loss_feature[0], dim=[2, 3])
            sorted_index = torch.argsort(mae, descending=True)
        elif sort_type == 'cos_similarity':
            _input = F.interpolate(_input, weight_shape[-2:]).detach()
            _input_feat = _input.view(weight_shape[0], _input.size()[1], -1)
            feat_maps = self.loss_feature[0].view(weight_shape[0], weight_shape[1], -1)
            cossim = torch.cosine_similarity(feat_maps, _input_feat, dim=2)
            sorted_index = torch.argsort(cossim, descending=True)
        elif sort_type == 'channel_mean':
            channel_mean = torch.mean(torch.abs(self.loss_feature[0]), dim=[1, 2,3], keepdim=True)
            large_feat = (torch.abs(self.loss_feature[0]) >= channel_mean).sum(dim=[2,3])
            sorted_index = torch.argsort(large_feat, dim=1, descending=True)
        elif sort_type == "nonzero":
            non_zero = torch.count_nonzero(self.loss_feature[0] == 0, dim=[2, 3])
            sorted_index = torch.argsort(non_zero, dim=1, descending=True)
        elif sort_type == "gradient":
            non_zero = torch.count_nonzero(weights_max != 0, dim=[2, 3])
            sorted_index = torch.argsort(non_zero, dim=1, descending=True)
        else:
            sorted_index = np.empty(self.loss_feature[0].size()[:2])
            for i in range(self.loss_feature[0].size()[0]):
                idx = list(range(0, self.loss_feature[0].size()[1]))
                random.shuffle(idx)
                sorted_index[i, :] = idx
            sorted_index = torch.LongTensor(sorted_index).to(_input.device)


        grad_front = []
        grad_rear = []
        index = 0
        for feat, grad, idx in zip(self.loss_feature[0], weights_max, sorted_index):
            idx_front = idx[:len(idx)//2]
            idx_rear = idx[len(idx)//2:]
            if kwargs['feat_type'] == 'half_feat_and_grad':
                grad_front.append(grad[idx_front] * feat[idx_front])
                grad_rear.append(grad[idx_rear] * feat[idx_rear])
            elif kwargs['feat_type'] == 'half_feat':
                grad_front.append(feat[idx_front])
                grad_rear.append(feat[idx_rear])
            index += 1
        if kwargs['feat_type'] == 'half_feat_and_grad' or kwargs['feat_type'] == 'half_feat':
            significant_feature = torch.stack(grad_front, dim=0)
            less_significant_feature = torch.stack(grad_rear, dim=0)
        if kwargs['loss_type'] == 'abs':
            if kwargs['feat_type'] == 'all_feat_and_grad':
                loss_to_minimum = torch.log(torch.sum(torch.abs(significant_feature)) / 2)
                loss_to_maximum = torch.log(torch.sum(torch.abs(less_significant_feature)) / 2)
            elif kwargs['feat_type'] == 'all_feat':
                loss_to_minimum = torch.log(torch.sum(torch.abs(self.loss_feature[0])) / 2)
                # loss_to_maximum = torch.log(torch.sum(torch.abs(less_significant_feature)) / 2)
            else:
                loss_to_minimum = torch.log(torch.sum(torch.abs(significant_feature)) / 2)
                loss_to_maximum = torch.log(torch.sum(torch.abs(less_significant_feature)) / 2)
        elif kwargs['loss_type'] == 'square':
            if kwargs['feat_type'] == 'all_feat_and_grad':
                loss_to_minimum = torch.log(torch.sum(torch.square(self.loss_feature[0] * weights_max)) / 2)
                # loss_to_maximum = torch.log(torch.sum(torch.square(less_significant_feature)) / 2)
            elif kwargs['feat_type'] == 'all_feat':
                loss_to_minimum = torch.log(torch.sum(torch.square(self.loss_feature[0])) / 2)
                # loss_to_maximum = torch.log(torch.sum(torch.square(less_significant_feature)) / 2)
            else:
                loss_to_minimum = torch.log(torch.sum(torch.square(significant_feature)) / 2)
                loss_to_maximum = torch.log(torch.sum(torch.square(less_significant_feature)) / 2)


        if kwargs['feat_type'] == 'all_feat' or kwargs['feat_type'] == 'all_feat_and_grad':
            loss = loss_to_minimum
        else:
            loss = loss_to_minimum - 1. * loss_to_maximum

        del self.loss_feature
        return output, loss

def plot_feat(feature, filename):
    fig = plt.figure(figsize=(20, 40))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.8, hspace=0, wspace=0.2)
    for i in range(128):
        ax = fig.add_subplot(16, 8, i + 1, xticks=[], yticks=[])
        ax.imshow(feature[i].cpu().data.numpy(), cmap="gray")
    plt.show()
    fig.savefig(f'save_images/{filename}.png')
