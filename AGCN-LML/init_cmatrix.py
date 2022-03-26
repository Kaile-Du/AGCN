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




def init_cmatrix():
    # 共现矩阵 
    adj_matrix = np.zeros(shape=(num_classes,num_classes))

    nums_matrix = np.zeros(shape=num_classes)

    for i in range(num_classes):
        nums_matrix[i] = 0.000001
    adj = {'adj': adj_matrix,
           'nums': nums_matrix}
    pickle.dump(adj, open('.../adj' + str(task_id) + '.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)

def init_hh_cmatrix(task_id):
    # 共现矩阵
    adj_matrix = np.zeros(shape=(num_classes,num_classes))
    nums_matrix = np.zeros(shape=num_classes)
    for i in range(num_classes):
        nums_matrix[i] = 0.000001

    adj = {'adj': adj_matrix,
           'nums': nums_matrix}
    pickle.dump(adj, open('.../hh_adj' + str(task_id) + '.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)



def init_sh_cmatrix(task_id):
    task_id = str(task_id)
    # 共现矩阵
    adj_matrix = np.zeros(shape=(int(task_id) * num_classes, num_classes))
    hard_sum_every_class = np.zeros(shape=num_classes)
    soft_sum_every_class = np.zeros(shape=int(task_id) * num_classes)
    for i in range(num_classes):
        hard_sum_every_class[i] = 0.0000000001
    for j in range(int(task_id) * num_classes):
        soft_sum_every_class[j] = 0.0000000001
    soft_or_hard_sum = 0.0000000001

    adj = {'adj': adj_matrix,
           'nums': hard_sum_every_class,
           's_or_h_sum': soft_or_hard_sum,
           's_sum_every_class': soft_sum_every_class}
    pickle.dump(adj, open('.../sh_adj' + task_id + '.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
