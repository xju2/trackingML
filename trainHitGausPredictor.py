#!/usr/bin/evn python


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

import pickle

from utils import tunable_parameters
from models import HitGausPredictor
from train import train_model
from loss_funs import gaus_llh_loss

from optparse import OptionParser


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

track_list_raw = pickle.load(open('ten_hists.npy', 'rb'))

mean_r, sigma_r = 913.681763, 692.430542
mean_phi, sigma_phi = 0.009939, 1.823752
mean_z, sigma_z = -2.315056, 1061.912476
for event in track_list_raw:
    for tracks in event:
        tracks[:, 0] = (tracks[:, 0] - mean_r)/sigma_r
        tracks[:, 1] = (tracks[:, 1] - mean_phi)/sigma_phi
        tracks[:, 2] = (tracks[:, 2] + mean_z)/sigma_z

track_arrays = np.concatenate([np.array(x) for x in track_list_raw])
print("total number of tracks: {}, with {} hits".format(
      track_arrays.shape[0], track_arrays.shape[1]))

total = track_arrays.shape[0]
n_trains = int(total/2)
train_input = torch.from_numpy(track_arrays[0:n_trains, :-1, :]).to(device)
train_target = torch.from_numpy(track_arrays[0:n_trains, 1:, 1:]).to(device)


batch_size = 64
n_epochs = 50
model = HitGausPredictor(batch_size=batch_size, device=device).to(device)
optimizer = optim.Adam(model.parameters())
print('total tunable parameters:', tunable_parameters(model))

losses = train_model(model, optimizer, gaus_llh_loss,
                     train_input, train_target,
                     n_epochs, batch_size)

# save these information
torch.save(model.state_dict(), "model_hitGaus")

with open('loss_trainGauss.pkl', 'wb') as fp:
    pickle.dump(losses, fp)

