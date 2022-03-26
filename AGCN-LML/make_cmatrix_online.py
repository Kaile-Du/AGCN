import torch
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import numpy as np
import copy
import random
import csv
import codecs
from torch.nn import Parameter





num_classes = 4
def gen_P(A):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj
def gen_A3(t,_adj):
    _adj[_adj < t] = 0
    _adj[_adj >= t] = 1
    _adj = _adj * 0.25 / (_adj.sum(0, keepdims=True) + 1e-6)
    return _adj
def gen_A1(num_classes,adj_file):
    import pickle
    result = pickle.load(open(adj_file, 'rb'))
    _adj = result['adj']
    _nums = result['nums']

    for j in range(num_classes):
        for i in range(num_classes):
            _adj[i][j] = _adj[i][j] / _nums[j]
    return _adj
def gen_A1_soft(task_id,num_classes,adj_file):
    import pickle
    result = pickle.load(open(adj_file, 'rb'))
    _adj = result['adj']
    _nums = result['nums']

    for j in range(num_classes):
        for i in range(int(task_id)*num_classes):
            _adj[i][j] = _adj[i][j] / _nums[j]
    return _adj

def gen_A2(num_classes, t,_adj):
    _adj[_adj < t] = 0
    _adj[_adj >= t] = 1
    _adj = _adj * 0.25 / (_adj.sum(0, keepdims=True) + 1e-6)

    _adj = _adj + np.identity(num_classes, np.int) # 这句的作用就是加上对角线都为1，其余地方为0,大小为num_classes*num_classes的矩阵
    return _adj



def gen_A(num_classes, t, adj_file):
    import pickle
    result = pickle.load(open(adj_file, 'rb'))
    _adj = result['adj']
    _nums = result['nums']
    _nums = _nums[:, np.newaxis]
    _adj = _adj / _nums
    _adj[_adj < t] = 0
    _adj[_adj >= t] = 1
    _adj = _adj * 0.25 / (_adj.sum(0, keepdims=True) + 1e-6)
    _adj = _adj + np.identity(num_classes, np.int)
    return _adj

#gen_adj()相当于通过A得到A_hat矩阵
def gen_adj(A):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj


def online_update_hh_cmatrix(image_ids,task_id):
    task_id = str(task_id)
    adj_file = '.../hh_adj'+task_id+'.pkl'
    train_csv = '.../data/task' + str(task_id) + '//train.csv'
    result = pickle.load(open(adj_file, 'rb'))
    adj_matrix = result['adj']
    nums_matrix = result['nums']
    dataset = pd.read_csv(train_csv)
    dataset = dataset.values
    dataset = dataset.tolist()

    for p in range(len(image_ids)):
        for index in range(len(dataset)):
            if(dataset[index][0] == image_ids[p]):
                data = dataset[index]
                for i in range(num_classes):
                    if data[i + 1] == 1:
                        nums_matrix[i] += 1
                        for j in range(num_classes):
                            if j != i:
                                if data[j + 1] == 1:
                                    adj_matrix[i][j] += 1
    # print('hh',adj_matrix)
    adj = {'adj': adj_matrix,
           'nums': nums_matrix}
    # print('online_update_cmatrix',adj_matrix)
    pickle.dump(adj, open('.../hh_adj'+task_id+'.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
    # print('##finish##')


def online_update_sh_cmatrix(image_ids,task_id, dist_targets):
    # print('online_update_sh_cmatrix is working')
    adj_file = '.../sh_adj' + str(task_id) + '.pkl'
    train_csv = '.../data/task' + str(task_id) + '//train.csv'
    result = pickle.load(open(adj_file, 'rb'))

    adj_matrix = result['adj']
    hard_sum_every_class = result['nums']
    soft_or_hard_sum = result['s_or_h_sum']
    soft_sum_every_class = result['s_sum_every_class']

    dataset = pd.read_csv(train_csv)
    dataset = dataset.values
    dataset = dataset.tolist()
    # image_ids = image_ids.tolist()
    dist_targets = dist_targets.tolist()
    # print('image_ids',image_ids)
    for p in range(len(image_ids)):  # len(image_ids)

        # print('p is working')

        soft_label = dist_targets[p]

        '''
        print('soft_label',soft_label)
        '''

        # for j in range(num_classes):
        #     # nums_matrix_soft[j] 表示第j类的软标签相加
        #     soft_sum = soft_sum + soft_label[j]
        for index in range(len(dataset)):
            if (dataset[index][0] == image_ids[p]):
                hard_label = dataset[index]
                soft_or_hard_sum += 1
                for g in range(int(task_id) * num_classes):
                    soft_sum_every_class[g] += soft_label[g]
                for y in range(num_classes):
                    if hard_label[y + 1] == 1:
                        hard_sum_every_class[y] += 1
                        for j in range(int(task_id)*num_classes):
                            adj_matrix[j][y] += soft_label[j]

    adj = {'adj': adj_matrix,
           'nums': hard_sum_every_class,
           's_or_h_sum': soft_or_hard_sum,
           's_sum_every_class': soft_sum_every_class}

    pickle.dump(adj, open('.../sh_adj' + str(task_id) + '.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)


def generate_aug_cmatrix(task_id,index):
    t = 0.4
    t2 = 0.3
    P_aug = np.zeros(shape=(num_classes * (int(task_id) + 1), num_classes * (int(task_id) + 1)))
    P_aug_full = np.zeros(shape=(num_classes * (int(task_id) + 1), num_classes * (int(task_id) + 1)))

    adj_file = '.../P_aug_full' + str(task_id - 1) + '.pkl'
    result = pickle.load(open(adj_file, 'rb'))
    pre_P_aug_full = result['P']

    adj_file = '.../P_aug' + str(task_id - 1) + '.pkl'
    result = pickle.load(open(adj_file, 'rb'))
    pre_P_aug = result['P']
    ############################soft-hard hard-soft###########################################################
    adj_file = '.../sh_adj' + str(task_id) + '.pkl'
    P_sh_matrix = gen_A1_soft(task_id,num_classes, adj_file)
    result = pickle.load(open(adj_file, 'rb'))
    hard_sum_every_class = result['nums']
    soft_sum_every_class = result['s_sum_every_class']
    soft_or_hard_sum = result['s_or_h_sum']


    P_hard = [0] * num_classes
    P_soft = [0] * int(task_id)*num_classes
    for h in range(num_classes):
        P_hard[h] = hard_sum_every_class[h] / soft_or_hard_sum
    for k in range(int(task_id)*num_classes):
        P_soft[k] = soft_sum_every_class[k] / soft_or_hard_sum


    P_hs_matrix = np.zeros(shape=(num_classes, int(task_id)*num_classes))
    for i in range(int(task_id*num_classes)):
        for j in range(int(num_classes)):
            a = P_sh_matrix[i][j]
            b = P_hard[j]
            c = P_soft[i]
            P_hs_matrix[j][i] = (a * b / c)/2
            # print('P_hs_matrix[j][i]', P_hs_matrix[j][i])
            if (P_hs_matrix[j][i] > 1):
                print('0000000000000000000')
                print('P_sh_matrix[i][j]',P_sh_matrix[i][j])
                print('P_hard[j]',P_hard[j])
                print('P_soft[j]',P_soft[j])
                print('P_hs_matrix[j][i]',P_hs_matrix[j][i])

    for i in range(task_id*num_classes):
        for j in range(num_classes):
            P_aug_full[i][j+task_id*num_classes] = P_sh_matrix[i][j]
            P_aug_full[j+task_id*num_classes][i] = P_hs_matrix[j][i]

    P_sh_matrix = gen_A3(t2, P_sh_matrix)
    P_sh_matrix = Parameter(torch.from_numpy(P_sh_matrix).float())

    P_hs_matrix = gen_A3(t2, P_hs_matrix)
    P_hs_matrix = Parameter(torch.from_numpy(P_hs_matrix).float())

    ################ hard and soft label#########################
    for i in range(task_id*num_classes):
        for j in range(num_classes):
            P_aug[i][j+task_id*num_classes] = P_sh_matrix[i][j]
            P_aug[j+task_id*num_classes][i] = P_hs_matrix[j][i]
    ############################soft-hard hard-soft###########################################################

    ############################soft-soft###########################################################
    for i in range(num_classes * task_id):
        for j in range(num_classes * task_id):
            P_aug[i][j] = pre_P_aug[i][j]

    for i in range(num_classes * task_id):
        for j in range(num_classes * task_id):
            P_aug_full[i][j] = pre_P_aug_full[i][j]
    ############################soft-soft###########################################################

    ############################hard-hard###########################################################
    adj_file = '.../hh_adj' + str(task_id) + '.pkl'
    P_hh_matrix = gen_A1(num_classes, adj_file)
    P_hh_matrix = P_hh_matrix + np.identity(num_classes, np.int)
    # P_hh_matrix = gen_A2(num_classes, t, P_hh_matrix)
    P_hh_matrix = Parameter(torch.from_numpy(P_hh_matrix).float())
    # P_hh_matrix = gen_P(P_hh_matrix)
    for i in range(task_id * num_classes, (task_id + 1) * num_classes):
        for j in range(task_id * num_classes, (task_id + 1) * num_classes):
            P_aug_full[i][j] = P_hh_matrix[i - task_id * num_classes][j - (task_id + 1) * num_classes]



    adj_file = '.../hh_adj' + str(task_id) + '.pkl'
    P_hh_matrix = gen_A1(num_classes,adj_file)
    # print('1 P_hh_matrix',P_hh_matrix)
    P_hh_matrix = gen_A2(num_classes, t, P_hh_matrix)
    # print('2 afterA2 P_hh_matrix',P_hh_matrix)
    P_hh_matrix = Parameter(torch.from_numpy(P_hh_matrix).float())
    P_hh_matrix = gen_P(P_hh_matrix)
    # print('3 aftergen_P P_hh_matrix', P_hh_matrix)
    for i in range(task_id * num_classes, (task_id + 1) * num_classes):
        for j in range(task_id * num_classes, (task_id + 1) * num_classes):
            P_aug[i][j] = P_hh_matrix[i - task_id * num_classes][j - (task_id + 1) * num_classes]
    ############################hard-hard###########################################################
    P_aug = Parameter(torch.from_numpy(P_aug).float())
    P = {'P': P_aug}
    pickle.dump(P, open('.../P_aug' + str(task_id) + '.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
    P_aug_full = Parameter(torch.from_numpy(P_aug_full).float())
    P_full = {'P': P_aug_full}
    pickle.dump(P_full, open('.../P_aug_full' + str(task_id) + '.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)

    return P_aug
