#!coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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

            if torch.isnan(loss):
                print("Loss of batch {} is NAN".format(ibatch))
                break
            sum_loss += loss.item()

        loss_train.append(sum_loss/nbatches)
    return loss_train


def gaus_llh_loss(outputs, targets):
    """Custom gaussian log-likelihood loss function"""
    if torch.isnan(outputs).any():
        raise Exception("Net's output is NAN")
    batches = outputs.size(0)
    hits = outputs.size(1)

    # Flatten layer axis into batch axis to use batch matrix operations
    outputs = outputs.contiguous().view(-1, outputs.size(-1))
    targets = targets.contiguous().view(-1, targets.size(-1))

    # Calculate the residual error
    dx1 = outputs[:, 0] - targets[:, 0]
    dx2 = outputs[:, 1] - targets[:, 1]
    c1 =  outputs[:, 2]
    c2 =  outputs[:, 3]
    rho = outputs[:, 4]

    det_sigma = (1 - rho*rho) * c1 * c2
    log_det = torch.log(det_sigma)
    chi2 = (dx1*dx1/c1 + dx2*dx2/c2 - 2*rho*dx1*dx2/torch.sqrt(c1*c2))/(1-rho*rho)
    #prob = log_det + chi2
    prob = torch.sqrt(det_sigma) + chi2

    return torch.sum(prob)/batches/hits


class HitGausPredictor(nn.Module):
    """
    A PyTorch module for particle track state estimation and hit prediction.

    This module is an RNN which takes a sequence of hits and produces a
    Gaussian shaped prediction for the location of the next hit.
    """

    def __init__(self, hidden_dim=5, batch_size=64, device=None, with_correlation=False):
        super(HitGausPredictor, self).__init__()
        input_dim = 3
        output_dim = 2
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        out_size = 5

        self.fc = nn.Linear(hidden_dim, out_size)
        self.device = device
        self.batch_size = batch_size


    def forward(self, x):
        """Might want to accept also the radius of the target layer."""
        input_size = x.size()

        # Initialize the LSTM hidden state
        h = (torch.zeros(self.lstm.num_layers, self.batch_size,
                         self.lstm.hidden_size, device=self.device),
             torch.zeros(self.lstm.num_layers, self.batch_size,
                         self.lstm.hidden_size, device=self.device))

        # Apply the LSTM module
        x, h = self.lstm(x, h)
        # Squash layer axis into batch axis
        x = x.contiguous().view(-1, x.size(-1))

        # Apply linear layer
        output = self.fc(x)
        output[:, 2] = torch.exp (output[:, 2]) # ensure it's positive
        output[:, 3] = torch.exp (output[:, 3]) # ensure it's positive
        output[:, 4] = torch.tanh(output[:, 4]) # ensure it ranges from [-1, 1]

        output = output.contiguous().view(input_size[0], input_size[1], -1)
        return output
