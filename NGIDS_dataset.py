import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class NGIDS_Dataset(torch.utils.data.Dataset):
    def __init__(self, data, label, p2i, s2i):
       
        self.data = []
        for i in data :

            path_l = []
            sys_l = []

            path, sys = i

            for p in path :
                path_l.append(p2i[p])

            for s in sys :
                sys_l.append(s2i[s])

            self.data.append(list(zip(path_l, sys_l)))
        
        self.data = np.array(self.data)
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i], self.label[i]


def NGIDS_get(slide_window_size, label_op = 1, NGIDS_path = './dataset/NGIDS_host_log_1-99.csv'):

    NGIDS = pd.read_csv(NGIDS_path)
            
    dropna_NGIDS = NGIDS.dropna(subset=['path', 'sys_call', 'label'])

    path = np.array(dropna_NGIDS['path'].to_list())
    syscall = np.array(dropna_NGIDS['sys_call'].to_list())
    label = np.array(dropna_NGIDS['label'].to_list())

    l = int(len(path) / slide_window_size)

    path = path[:l * slide_window_size].reshape(l, slide_window_size)
    syscall = syscall[:l * slide_window_size].reshape(l, slide_window_size)
    label = label[:l * slide_window_size].reshape(l, slide_window_size)

    if label_op == 1 :
        label = np.max(label, axis = 1)
    else :
        label = label[:, -1]

    positive_path = []
    positive_syscall = []

    negative_path = []
    negative_syscall = []

    for i in range(l) :
        if label[i] == 1 :
            negative_path.append(path[i])
            negative_syscall.append(syscall[i])
        else :
            positive_path.append(path[i])
            positive_syscall.append(syscall[i])


    positive_len = len(positive_path)
    negative_len = len(negative_path)

    print("positive : ", positive_len)
    print("negative : ", negative_len)

    X_train, X_vali, y_train, y_vali = train_test_split(
    list(zip(positive_path[:positive_len - negative_len - 1], positive_syscall[:positive_len - negative_len - 1]))
    , [0 for i in range(positive_len - negative_len - 1)], test_size=0.2, random_state=42)

    X_test = list(zip(positive_path[positive_len - negative_len : positive_len] + negative_path, 
                    positive_syscall[positive_len - negative_len : positive_len] + negative_syscall))
    y_test = [ 0 for i in range(negative_len)] + [ 1 for i in range(negative_len)]


    return X_train, y_train, X_vali, y_vali, X_test, y_test
