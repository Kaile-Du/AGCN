# -*- coding: utf-8 -*-
"""
"""
# License: BSD
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import sys
import copy

from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from evaluation_metrics import prf_cal, cemap_cal
import scipy
import scipy.io
from myresnet_fc import resnet101
import pdb

from torch.nn import Parameter
import pickle
from GCNRSN import gnn
from GCNRSN import GraphConvolution
from make_cmatrix_online import online_update_hh_cmatrix, online_update_sh_cmatrix, generate_aug_cmatrix
from init_cmatrix import init_hh_cmatrix, init_sh_cmatrix
from torch.nn import Parameter
from skimage import io, transform


trial = 2
img_size = 448
MAP = 0
OP = 0
OR = 0
OF1 = 0
CP = 0
CR = 0
CF1 = 0


def gen_A1(num_classes,adj_file):
    import pickle
    result = pickle.load(open(adj_file, 'rb'))
    _adj = result['adj']
    _nums = result['nums']
    for j in range(num_classes):
        for i in range(num_classes):
            _adj[i][j] = _adj[i][j] / _nums[j]
    return _adj

def gen_A2(num_classes, t, _adj):
    _adj[_adj < t] = 0
    _adj[_adj >= t] = 1
    _adj = _adj * 0.28 / (_adj.sum(0, keepdims=True) + 1e-6)
    _adj = _adj + np.identity(num_classes, np.int)
    return _adj


def gen_P(A):
    D = torch.pow(A.sum(1).float(), -0.8)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj


trial = 2
img_size = 448
"""
if len(sys.argv)>=2:
    trial = int(sys.argv[1])
if len(sys.argv)>=3:
    img_size = int(sys.argv[2])
"""
randseed = trial
np.random.seed(randseed)
torch.manual_seed(randseed)

class cocoDataset(Dataset):
    def __init__(self, label_file, image_dir, transform=None):
        self.labels = pd.read_csv(label_file, header=None)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir,
                                self.labels.iloc[idx, 0])
        image = io.imread(img_name)
        label = self.labels.iloc[idx, 1:].values
        image_id = self.labels.iloc[idx, :].values
        label = label.astype('double')
        if len(image.shape) == 2:
            image = np.expand_dims(image, 2)
            image = np.concatenate((image, image, image), axis=2)
        if self.transform:
            image = self.transform(image)
        # id = torch.Tensor(image_id[0])
        return image_id[0],image, label



# trainval需要水平翻转，val与test不需要水平翻转
data_transforms = {
    'train': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.488, 0.486, 0.406], [0.229, 0.224, 0.228])
    ]),
    'val': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.488, 0.486, 0.406], [0.229, 0.224, 0.228])
    ]),
    'test': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.488, 0.486, 0.406], [0.229, 0.224, 0.228])
    ])
}



def train_model(inp, CNN, GNN, criterion, dist_criterion, optimizer_CNN, optimizer_GNN, task_id,
          pre_CNN=None, num_epochs=1):
    since = time.time()
    t = 0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        print('Current learning rate: ' + '%.8f' % 0.0001)
        CNN.train()
        GNN.train()
        running_loss = 0.0
        for index, (image_ids, inputs, labels) in enumerate(train_loader):
            online_update_hh_cmatrix(image_ids, task_id)
            if (task_id == 0):
                adj_file = '.../hh_adj0.pkl'
                P_hh_matrix = gen_A1(num_classes, adj_file)
                P_t = P_hh_matrix
                P_t = P_t + np.identity(num_classes, np.int)
                P_full = {'P': P_t}
                if index == 0:
                    pickle.dump(P_full, open('.../P_task' + str(task_id) + '_start' + '.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
                if index == 725:
                    pickle.dump(P_full, open('.../P_task' + str(task_id) + '_middle' + '.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
                pickle.dump(P_full, open('.../P_aug_full' + str(task_id) + '.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
                P_aug = gen_A2(num_classes, t, P_hh_matrix)
                P_aug = Parameter(torch.from_numpy(P_aug).float())
                P_aug = gen_P(P_aug)
                P = {'P': P_aug}

                pickle.dump(P, open('.../P_aug' + str(task_id) + '.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
                P = P_aug
            inputs = inputs.to(device)
            labels = labels.to(device)

            inputs = inputs.float()
            labels = labels.float()

            # zero the parameter gradients
            optimizer_CNN.zero_grad()
            optimizer_GNN.zero_grad()

            # forward
            with torch.set_grad_enabled(True):
                if task_id > 0:
                    pre_feature = pre_CNN(inputs)
                    pre_h = pre_GNN(pre_P, pre_inp)
                    pre_feature = pre_feature.view(pre_feature.size(0), -1)
                    pre_l_GCN = torch.matmul(pre_feature, pre_h)
                    dist_targets_GCN = torch.sigmoid(pre_l_GCN)
                    Softmax = nn.Softmax(dim=1)
                    dist_targets_GCN_s = Softmax(pre_l_GCN)

                    online_update_sh_cmatrix(image_ids,task_id, dist_targets_GCN_s)
                    P = generate_aug_cmatrix(task_id,index)
                # 计算分类损失
                ########################################################################################

                feature = CNN(inputs)
                start = (task_id) * 4
                end = (task_id + 1) * 4

                h = GNN(P, inp)
                l_GCN = torch.matmul(feature, h)


                loss = criterion(l_GCN[:, start:end], labels)
                ########################################################################################


                # 计算蒸馏损失
                ########################################################################################
                if (task_id > 0):

                    dist_targets_GNN_ = pre_h

                    dist_loss_GCN = dist_criterion(l_GCN[:, 0:(task_id) * 4], dist_targets_GCN)

                    dist_targets_GNN = dist_targets_GNN_.detach()
                    dist_pred_GNN = h[:, 0:(task_id) * 4]

                    dist_loss_GNN = gnn_criterion(dist_pred_GNN, dist_targets_GNN)


                ########################################################################################

                if task_id > 0:

                    loss = a * loss + b * dist_loss_GCN + dist_loss_GNN * c
                loss.backward(retain_graph=True)
                optimizer_CNN.step()

                optimizer_GNN.step()

            running_loss += loss.item() * inputs.size(0)


        epoch_loss = running_loss / dataset_sizes
        print('{} Loss: {:.4f}'.format(
            'train', epoch_loss))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    return CNN, GNN



def test_model(CNN, GNN, optimizer_CNN, optimizer_GNN, P, inp):
    since = time.time()
    CNN.eval()
    GNN.eval()
    for index, (image_ids, inputs, labels) in enumerate(test_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        inputs = inputs.float()
        labels = labels.float()
        optimizer_CNN.zero_grad()
        optimizer_GNN.zero_grad()

        with torch.set_grad_enabled(False):
            feature = CNN(inputs)
            h = GNN(P, inp)
            l_GNN = torch.matmul(feature, h)
            outputs = l_GNN
            if index == 0:
                outputs_test = outputs
                labels_test = labels
            else:
                outputs_test = torch.cat((outputs_test, outputs), 0)
                labels_test = torch.cat((labels_test, labels), 0)
    mAP, emap = cemap_cal(outputs_test.to(torch.device("cpu")).numpy(), labels_test.to(torch.device("cpu")).numpy())
    print('Test:')
    print(mAP)

    OP, OR, OF1, CP, CR, CF1 = prf_cal(outputs_test.to(torch.device("cpu")).numpy(),
                                       labels_test.to(torch.device("cpu")).numpy())

    return OP, OR, OF1, CP, CR, CF1, mAP



num_tasks = 10
num_classes = 4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
bs = 8
print('###batch_size###', bs)
t = 0.4
print('t1', t)
print('t2', 0.3)
test_results = dict()

for task_id in range(num_tasks):

    relu = nn.ReLU(inplace=False)
    init_hh_cmatrix(task_id)
    if task_id > 0:
        init_sh_cmatrix(task_id)

    task_id = str(task_id)
    print('#################task' + task_id + '##################:')
    train_datasets = cocoDataset('.../data/task' + task_id + '//train.csv',
                                 '.../MSCOCO/train2014', data_transforms['train'])
    train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=bs, shuffle=True, num_workers=4)
    dataset_sizes = len(train_datasets)

    inp_name = '.../word' + task_id + '/glove_wordEmbedding.pkl'
    with open(inp_name, 'rb') as f:
        inp = pickle.load(f)
    inp = torch.Tensor(inp)

    if (int(task_id) == 0):
        CNN = resnet101(pretrained=True, num_classes=8)
        line = nn.Linear(4096, num_classes)
        GNN = gnn(in_channel=300)
        pre_line = None
        pre_P = None
        pre_inp = None
        pre_CNN = None
        pre_GNN = None
        h0 = None
        h1 = None
        h2 = None
    else:
        pre_CNN = copy.deepcopy(CNN)
        pre_GNN = copy.deepcopy(GNN)


        pre_line = copy.deepcopy(line)

        in_features = line.in_features
        out_features = line.out_features
        weight = line.weight.data
        line = nn.Linear(in_features, out_features + num_classes)
        line.weight.data[:out_features] = weight


        adj_file = '.../P_aug' + str(int(task_id) - 1) + '.pkl'
        result = pickle.load(open(adj_file, 'rb'))
        P_ = result['P']
        pre_P = P_

        pre_inp_name = '.../word' + str(int(task_id) - 1) + '/glove_wordEmbedding.pkl'
        with open(pre_inp_name, 'rb') as f:
            pre_inp = pickle.load(f)
        pre_inp = torch.Tensor(pre_inp)

        if int(task_id) == 1:
            h0 = pre_GNN(pre_P, pre_inp)
        if int(task_id) == 2:
            h1 = pre_GNN(pre_P, pre_inp)
        if int(task_id) == 3:
            h2 = pre_GNN(pre_P, pre_inp)
        if int(task_id) == 4:
            h3 = pre_GNN(pre_P, pre_inp)
        if int(task_id) == 5:
            h4 = pre_GNN(pre_P, pre_inp)
        if int(task_id) == 6:
            h5 = pre_GNN(pre_P, pre_inp)
        if int(task_id) == 7:
            h6 = pre_GNN(pre_P, pre_inp)
        if int(task_id) == 8:
            h7 = pre_GNN(pre_P, pre_inp)
        if int(task_id) == 9:
            h8 = pre_GNN(pre_P, pre_inp)

    optimizer_CNN = optim.Adam(CNN.parameters(), lr=0.0001)
    optimizer_GNN = optim.Adam(GNN.parameters(), lr=0.00003)
    CNN = CNN.to(device)
    GNN = GNN.to(device)
    criterion = nn.MultiLabelSoftMarginLoss()
    dist_criterion = nn.MultiLabelSoftMarginLoss()
    gnn_criterion = nn.MSELoss()
    CNN, GNN = train_model(inp, CNN, GNN, criterion, dist_criterion, optimizer_CNN, optimizer_GNN,
                                int(task_id),pre_CNN=pre_CNN, num_epochs=1)
    torch.save(CNN.state_dict(), '.../CNN_model' + str(task_id) + '.pt')
    torch.save(GNN.state_dict(), '.../GNN_model' + str(task_id) + '.pt')

    adj_file = '.../P_aug' + str(task_id) + '.pkl'
    result = pickle.load(open(adj_file, 'rb'))
    P = result['P']


###############################################################################################################

    print('task' + task_id + 'performance:')

    test_results[int(task_id)] = []
    for i in range((int(task_id)+1)):
        test_datasets = cocoDataset('.../data/task' + str(i) + '/continuous/test.csv',
                                    '.../MSCOCO/val2014', data_transforms['test'])
        test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=bs, shuffle=False)

        adj_file = '.../P_aug' + str(i) + '.pkl'
        result = pickle.load(open(adj_file, 'rb'))
        P = result['P']

        inp_name = '.../word' + str(i) + '/glove_wordEmbedding.pkl'
        with open(inp_name, 'rb') as f:
            inp = pickle.load(f)
        inp = torch.Tensor(inp)

        OP, OR, OF1, CP, CR, CF1, MAP = test_model(CNN, GNN, optimizer_CNN, optimizer_GNN, P, inp)
        test_results[int(task_id)].append([OP, OR, OF1, CP, CR, CF1, MAP])

    print("test_results", test_results)
    MAP_TASK_AVERAGE = 0
    for i in range((int(task_id)+1)):
        MAP_TASK_AVERAGE += test_results[int(task_id)][i][6]
    MAP_TASK_AVERAGE /= (int(task_id)+1)
    print("Task ",int(task_id))
    print("Map_AVERAGE",MAP_TASK_AVERAGE)

forget_list_7 = []
for i in range(num_tasks-1):
    forget_list_7.append([(test_results[i][i][0] - test_results[9][i][0])/test_results[i][i][0],
                          (test_results[i][i][1] - test_results[9][i][1])/test_results[i][i][1],
                          (test_results[i][i][2] - test_results[9][i][2])/test_results[i][i][2],
                          (test_results[i][i][3] - test_results[9][i][3])/test_results[i][i][3],
                          (test_results[i][i][4] - test_results[9][i][4])/test_results[i][i][4],
                          (test_results[i][i][5] - test_results[9][i][5])/test_results[i][i][5],
                          (test_results[i][i][6] - test_results[9][i][6])/test_results[i][i][6]])
    print("forget_list_7",forget_list_7[i])

forget_end = []
for i in range(7):
    forget_end.append((forget_list_7[0][i] + forget_list_7[1][i] + forget_list_7[2][i]+ forget_list_7[3][i]+ forget_list_7[4][i]+ forget_list_7[5][i]+ forget_list_7[6][i]+ forget_list_7[7][i]+ forget_list_7[8][i]) / 9)

forget_list_7_4 = []
for i in range(7):
    forget_list_7_4.append(
        (test_results[9][0][i] + test_results[9][1][i] + test_results[9][2][i] + test_results[9][3][i]+ test_results[9][4][i]+ test_results[9][5][i]+ test_results[9][6][i]+ test_results[9][7][i]+ test_results[9][8][i]+ test_results[9][9][i]) / 10)

print("forget Map:", forget_end[6])
print("forget CP:", forget_end[3])
print("forget CR:", forget_end[4])
print("forget CF1:", forget_end[5])
print("forget OP:", forget_end[0])
print("forget OR:", forget_end[1])
print("forget OF1:", forget_end[2])

print('\n')

print("Map:", forget_list_7_4[6])
print("CP:", forget_list_7_4[3])
print("CR:", forget_list_7_4[4])
print("CF1:", forget_list_7_4[5])
print("OP:", forget_list_7_4[0])
print("OR:", forget_list_7_4[1])
print("OF1:", forget_list_7_4[2])

