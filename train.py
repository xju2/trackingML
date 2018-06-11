#!/usr/bin/python

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

import logging
import pickle
import time
import sys
import os

# local modules
from process_data import data_uID

from utils import tunable_parameters
from utils import timeSince

from models import HitPredictor

def train(path='input/train_1/event000001000', n_iters=-1):
    data = data_uID()
    data.load_training(path)
    total_modules = data.modules

    criterion = nn.NLLLoss()
    rnn = HitPredictor(input_dim=total_modules+1, hidden_dim=20,
                   output_dim=total_modules+1,
                   n_lstm_layers=1)

    tmp_model_dir = 'tmp_trained_model'
    if os.path.exists(tmp_model_dir):
        rnn.load_state_dict(torch.load(tmp_model_dir))
        print("continue training")

    total_tunable = tunable_parameters(rnn)
    logging.info('total parameters in RNN: {}'.format(total_tunable))

    if n_iters < 0:
        n_iters = int(total_tunable*1.2)

    print("total iterations", n_iters)
    print_every = int(n_iters/50) + 1
    plot_every = int(n_iters/1000) + 1
    all_losses = []
    total_loss = 0
    start = time.time()

    optimizer = optim.SGD(rnn.parameters(), lr=0.01, momentum=0.9)

    for iter_ in range(n_iters):
        input_, target_ = data.random_input_target()

        # training
        rnn.zero_grad()
        rnn.hidden = rnn.init_hidden()


        output = rnn(input_)
        loss = criterion(output, target_)


        loss.backward()
        optimizer.step()

        normed_loss = loss.item()/input_.size(0)
        if normed_loss < 1E-3:
            break

        total_loss += normed_loss
        if iter_ % print_every == 0:
            print('%s (%d %d%%) %.4f' % (timeSince(start), iter_, iter_ / n_iters * 100, normed_loss))
            torch.save(rnn.state_dict(), tmp_model_dir)
            with open('tmp_loss_list', 'wb') as fp:
                pickle.dump(all_losses, fp)

            with torch.no_grad():
                #print("------------------------------ {}".format(iter_))
                #print("inputs:", torch.reshape(torch.argmax(input_, dim=2), (-1,)).numpy())
                #print("predictions:", torch.argmax(output, dim=1).numpy())
                logging.info("------------------------------ {}".format(iter_))
                logging.info("inputs: %s", np.array_str(torch.reshape(torch.argmax(input_, dim=2), (-1,)).numpy()))
                logging.info("predictions: %s", np.array_str(torch.argmax(output, dim=1).numpy()))

        if iter_ % plot_every == 0:
            all_losses.append(total_loss / plot_every)
            total_loss = 0

    torch.save(rnn.state_dict(), "final_trained_model")
    with open('loss_list', 'wb') as fp:
        pickle.dump(all_losses, fp)


if __name__ == "__main__":
    log_file_name = "log_01"
    logging.basicConfig(filename=log_file_name, level=logging.DEBUG,
                        format='<%(levelname)s> %(asctime)s : %(message)s',
                        datefmt='%Y/%m/%d %I:%M:%S %p'
                       )
    if len(sys.argv) > 1:
        train(sys.argv[1])
    else:
        train(n_iters=20)
