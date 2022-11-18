import pandas as pd
import numpy as np
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import NGIDS_dataset

def run(batch_size, slide_window_size, learning_rate, max_epochs, hidden_size, hhidden_size, latent_vector, early_stop, vector_size, window, model_cont="GRU"):

    device = torch.device('cuda') # GPU 사용

    X_train, y_train, X_vali, y_vali, X_test, y_test = NGIDS_dataset.NGIDS_get(slide_window_size)

    print(len(X_train), len(y_train))
    print(len(X_vali), len(y_vali))
    print(len(X_test), len(y_test))

    import gensim

    def save_path(vector_size, window, data_name="NGIDS_path_w2v"):
        return "./dataset/PathSystem/" + f"vectorsize{vector_size}_window{window}_" + data_name

    def save_sys(vector_size, window, data_name = "NGIDS_vector"):
        return "./dataset/PathSystem/" + f"vectorsize{vector_size}_window{window}_" + data_name

    input_size = vector_size

    NGIDS_sys_model = gensim.models.Word2Vec.load(save_sys(vector_size, window))
    NGIDS_path_model = gensim.models.Word2Vec.load(save_path(vector_size, window, "NGIDS_vector"))

    p2i = NGIDS_path_model.wv.key_to_index
    s2i = NGIDS_sys_model.wv.key_to_index

    NGIDS_trainset = NGIDS_dataset.NGIDS_Dataset(X_train, y_train, p2i, s2i)
    train_loader = DataLoader(NGIDS_trainset, batch_size=batch_size, shuffle = True)

    NGIDS_valiset = NGIDS_dataset.NGIDS_Dataset(X_vali, y_vali, p2i, s2i)
    vali_loader = DataLoader(NGIDS_valiset, batch_size=batch_size, shuffle = True)

    NGIDS_testset = NGIDS_dataset.NGIDS_Dataset(X_test, y_test, p2i, s2i)

    import models
    from models import GRU_Repeat_AutoEncoder, CNN_AutoEncoder

    #model = GRU_Repeat_AutoEncoder(input_size, hidden_size, hhidden_size, latent_vector, num_layers, NGIDS_path_model.wv.vectors, NGIDS_sys_model.wv.vectors, device)

    model = CNN_AutoEncoder(input_size, hidden_size, hhidden_size, latent_vector, NGIDS_path_model.wv.vectors, NGIDS_sys_model.wv.vectors, device)
    model.to(device)
    

    model, loss = models.run(model, train_loader, vali_loader, learning_rate, max_epochs, early_stop)

    path = f'./result/{model_cont}/s{slide_window_size}h{hidden_size}hh{hhidden_size}l{latent_vector}/v{vector_size}w{window}/'

    torch.save(model, path + "AutoEncoder.model")
    torch.save(NGIDS_trainset, path + "trainset")
    torch.save(NGIDS_valiset, path + "valiset")
    torch.save(NGIDS_testset, path + "testset")
    torch.save(loss, path + "loss")