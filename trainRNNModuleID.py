#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from trackml.dataset import load_event
from trackml.randomize import shuffle_hits
from trackml.score import score_event

import pickle
import os
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device: ", device)

from rnn_module_id import ModuleIDBasedRNN
from rnn_module_id import load_training

from utils import tunable_parameters
from rnn_module_id import modules

train_data = load_training()
print("Total events:", len(train_data))
print("Total truth particles:", sum([y.shape[0] for x,y in train_data]))


model = ModuleIDBasedRNN(input_dim=modules+1,
                         hidden_dim=20,
                         output_dim=modules+1,
                         batch_size=1,
                         device=device)

model.to(device)
print("total parameters:", tunable_parameters(model))

criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

loss_file_name = os.path.join('output', 'loss', 'ModuelRNN.pkg.gz')
training_mode = True
if training_mode:
    nepochs = 50
    all_losses = []

    for epoch in tqdm(range(nepochs)):
        # go through half of training data.
        loss_evt = []
        for ievt in tqdm(range(int(len(train_data)/2))):
            event, truth = train_data[ievt]
            total_loss = 0
            for pID in truth.values:
                hits = event[event['particle_id'] == pID]['uID'].values
                input_, target_ = input_target(hits)
                input_ = input_.to(device)
                target_ = target_.to(device)

                model.hidden = model.init_hidden()
                model.zero_grad()
                output = model(input_)
                loss = criterion(output, target_)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()/input_.size(1)

            loss_evt.append(total_loss/truth.shape[0])
        all_losses.append(loss_evt)
        torch.save(model.state_dict(),
                   os.path.join('output', 'model', 'RNNModule_'+str(epoch))
                  )
        with open(os.path.join('output', 'loss', 'RNNModule_'+str(epoch)), 'wb') as fp:
            pickle.dum(loss_evt, fp)

    with open(loss_file_name, 'wb') as fp:
        pickle.dump(all_losses, fp)
else:
    # in testing mode
    with open(loss_file_name, 'rb') as fp:
        all_losses = pickle.load(fp)

    print(all_losses)
