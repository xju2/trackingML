#!/usr/bin/python

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import pandas as pd

import logging
import pickle
import time
import sys
import os
from tqdm import tqdm

sys.path.insert(0, '../common')


from utils import tunable_parameters
from utils import timeSince
from utils import device

from process_data import get_real_tracks
from process_data import get_fake_tracks
from process_data import random_inputs

from is_good_track import IsGoodTrack

from optparse import OptionParser

def train(options):
    event = options.path
    n_epochs = options.iters

    real_tracks_out_name = os.path.join(options.output, 'reak_tracks_pad.pkl')
    fake_tracks_out_name = os.path.join(options.output, 'fake_tracks_pad.pkl')

    if os.path.exists(real_tracks_out_name):
        real_tracks = pickle.load(open(real_tracks_out_name, 'rb'))
        fake_tracks = pickle.load(open(fake_tracks_out_name, 'rb'))
    else:
        real_tracks = get_real_tracks(event)
        fake_tracks = get_fake_tracks(event, real_tracks.shape[0])
        # save the two dataset
        with open(fake_tracks_out_name, 'wb') as fp:
            pickle.dump(fake_tracks, fp)

        with open(real_tracks_out_name, 'wb') as fp:
            pickle.dump(real_tracks, fp)

    # check if GPU available
    print(device)

    # create RNN model
    input_dim = 60
    hidden_dim = 60
    batch_size = 64
    model = IsGoodTrack(input_dim, hidden_dim, batch_size, n_lstm_layers=3, device=device)
    print("total parameters:", tunable_parameters(model))

    model.to(device)

    criterion = nn.L1Loss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


    epoch_list = []
    loss_train = []
    loss_test = []

    for iter_ in tqdm(range(n_epochs)):
        sum_loss = 0
        epoch_list.append(iter_)
        icount = 0

        n_batches = int(real_tracks.shape[0]/batch_size)
        for nb in range(n_batches):

            model.hidden = model.init_hidden()
            model.zero_grad()

            input_track, target_score = random_inputs(real_tracks, fake_tracks, batch_size)
            #print(input_track.shape)
            input_tensor = torch.from_numpy(input_track).view(batch_size, 1, -1)
            target = torch.from_numpy(target_score).view(batch_size, -1)
            target = target.type(torch.FloatTensor)

            # send input and target to GPU
            input_tensor, target = input_tensor.to(device), target.to(device)

            output = model(input_tensor.view(batch_size, input_dim))

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()/batch_size

        loss_train.append(sum_loss/len(real_tracks))

    # save these information
    torch.save(model.state_dict(), os.path.join(options.output, "model_isgoodtrack"))
    with open(os.path.join(options.output, 'loss_train.pkl'), 'wb') as fp:
        pickle.dump(loss_train, fp)


if __name__ == "__main__":
    usage = "%prog [options]"
    version = '%prog 0.0.1'
    parser = OptionParser(usage=usage, description="train RNN with kaggle", version=version)

    parser.add_option('-i', '--iters', default=20, type=int, help='iterations')
    parser.add_option('-p', '--path', default='input/train_1/event000001000', help='directory for training data')
    parser.add_option('-o', '--outDir', default='output', help='output directory')

    (options, args) = parser.parse_args()

    if not os.path.exists(options.output):
        os.makedirs(options.output)

    log_file_name = os.path.join(options.output, "log_%s"%time.time())
    logging.basicConfig(filename=log_file_name, level=logging.DEBUG,
                        format='<%(levelname)s> %(asctime)s : %(message)s',
                        datefmt='%Y/%m/%d %I:%M:%S %p'
                       )
    print ("total epochs: ", options.iters)
    train(options)
