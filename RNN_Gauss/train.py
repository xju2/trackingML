#!/usr/bin/evn python


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

import pickle

import sys
sys.path.insert(0, '../common')

from utils import tunable_parameters
from utils import device

from hit_gauss_predictor import HitGausPredictor
from hit_gauss_predictor import train_model
from hit_gauss_predictor import gaus_llh_loss

from optparse import OptionParser

parser = OptionParser(description="train RNN with kaggle", version="0.1.0")
parser.add_option('-o', '--outName', default="hitGauss", help="prefix of output name")
parser.add_option('-d', '--outDir', default="output", help="output directory")
parser.add_option('-i', '--trainData', default="ten_hists_normed.npy", help="input training data")
parser.add_option('-b', '--batchSize', default=32, type=int, help="batch size")
parser.add_option('-e', '--epoch', default=100, type=int, help="number of epochs")

(options, args) = parser.parse_args()

import os
if not os.path.exists(options.trainData):
    print(options.trainData,"is missing.")
    exit(1)

if not os.path.exists(options.outDir):
    os.makedirs(options.outDir)

print("device: ", device)

track_arrays = pickle.load(open(options.trainData, 'rb'))
print("total number of tracks: {}, with {} hits".format(
      track_arrays.shape[0], track_arrays.shape[1]))

total = track_arrays.shape[0]
n_trains = int(total/2)
train_input = torch.from_numpy(track_arrays[0:n_trains, :-1, :]).to(device)
train_target = torch.from_numpy(track_arrays[0:n_trains, 1:, 1:]).to(device)


batch_size = options.batchSize
n_epochs = options.epoch

model = HitGausPredictor(batch_size=batch_size, device=device).to(device)

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
print('total tunable parameters:', tunable_parameters(model))

losses = train_model(model, optimizer, gaus_llh_loss,
                     train_input, train_target,
                     n_epochs, batch_size
                    )

# save these information
torch.save(model.state_dict(), os.path.join(options.outDir, options.outName+"_model"))

with open(os.path.join(options.outDir, options.outName+'_loss.pkl'), 'wb') as fp:
    pickle.dump(losses, fp)

print("DONE")
