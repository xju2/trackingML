#!/usr/bin/evn python


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

import pickle

from utils import tunable_parameters
from hit_gauss_predictor import HitGausPredictor
from train import train_model
from hit_gauss_predictor import gaus_llh_loss

from optparse import OptionParser


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device: ", device)

track_arrays = pickle.load(open('ten_hists_normed.npy', 'rb'))
print("total number of tracks: {}, with {} hits".format(
      track_arrays.shape[0], track_arrays.shape[1]))

total = track_arrays.shape[0]
n_trains = int(total/2)
train_input = torch.from_numpy(track_arrays[0:n_trains, :-1, :]).to(device)
train_target = torch.from_numpy(track_arrays[0:n_trains, 1:, 1:]).to(device)


batch_size = 128
n_epochs = 100
model = HitGausPredictor(batch_size=batch_size, device=device).to(device)
#optimizer = optim.Adam(model.parameters())
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
print('total tunable parameters:', tunable_parameters(model))

losses = train_model(model, optimizer, gaus_llh_loss,
                     train_input, train_target,
                     n_epochs, batch_size)

# save these information
torch.save(model.state_dict(), "model_hitGaus")

with open('loss_trainGauss.pkl', 'wb') as fp:
    pickle.dump(losses, fp)

