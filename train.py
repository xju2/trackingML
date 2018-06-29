#!/usr/bin/python

import torch
import numpy as np

from utils import tunable_parameters
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, optimizer, loss_func,
                input_arr, target_arr,
                n_epochs, batch_size):
    loss_train = []
    print("total entries: ", input_arr.size())
    n_training = input_arr.size(0)

    for i in tqdm(range(n_epochs)):
        sum_loss = 0.

        nbatches = int(n_training/batch_size)
        if i==0:
            print("number of batches:", nbatches)

        for ibatch in range(nbatches):
            start = ibatch*batch_size
            end = start + batch_size

            input_t = input_arr[start:end].to(device)
            target_t = target_arr[start:end].to(device)


            model.zero_grad()
            output_t = model(input_t)
            loss = loss_func(output_t, target_t)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()

        loss_train.append(sum_loss/nbatches)
    return loss_train
