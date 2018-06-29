# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class HitPredictor(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=20, output_dim=20,
                n_lstm_layers=1):
        super(HitPredictor, self).__init__()

        self.hidden_dim = hidden_dim
        self.batch_size = 1

        self.lstm = nn.LSTM(input_dim, hidden_dim, n_lstm_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def init_hidden(self):
        return (torch.zeros(self.lstm.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.lstm.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, x):
        output, self.hidden = self.lstm(x, self.hidden)
        output = self.fc(output.view(len(x), -1))
        output = self.dropout(output)
        tag_scores = self.softmax(output)
        return tag_scores


class IsGoodTrack(nn.Module):

    def __init__(self, input_dim=60, hidden_dim=120, batch_size=64, n_lstm_layers=1, device=None):

        super(IsGoodTrack, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_lstm_layers, batch_first=True, dropout=0.1)
        self.fc1 = nn.Linear(hidden_dim, int(input_dim/2))
        self.fc3 = nn.Linear(int(input_dim/2), 1)
        self.fc4 = nn.Linear(9, 1)
        self.sigma = nn.Sigmoid()
        self.device = device

        self.hidden_dim = hidden_dim

        self.batch_size = batch_size

    def init_hidden(self):
        return (torch.zeros(self.lstm.num_layers, self.batch_size, self.hidden_dim, device=self.device),
                torch.zeros(self.lstm.num_layers, self.batch_size, self.hidden_dim, device=self.device)
               )

    def forward(self, x):
        input_size = x.size()
        output, self.hidden = self.lstm(x.view(self.batch_size, 1, -1), self.hidden)
        output = self.fc1(output.view(self.batch_size, self.hidden_dim))
        output = self.sigma(output)
        output = self.fc3(output)
        output = self.sigma(output)
#         output = self.fc4(output)
#         output = self.sigma(output)
        return output

class HitGausPredictor(nn.Module):
    """
    A PyTorch module for particle track state estimation and hit prediction.

    This module is an RNN which takes a sequence of hits and produces a
    Gaussian shaped prediction for the location of the next hit.
    """

    def __init__(self, hidden_dim=5, batch_size=64, device=None):
        super(HitGausPredictor, self).__init__()
        input_dim = 3
        output_dim = 2
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
#         out_size = int(output_dim * (output_dim + 3) / 2)
        out_size = 4
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

        # Extract and transform the gaussian parameters
        means = output[:, :2]
        variances = output[:, 2:4] ## ensure it is positive
        variances = torch.exp(variances)

        # Expand the layer axis again, just for consistency/interpretability
        means = means.contiguous().view(self.batch_size, -1, 2)
        variances = variances.contiguous().view(self.batch_size, -1, 2)
        return means, variances


def cal_res(test_track):
    test_t = torch.from_numpy(test_track)
    target_t = torch.from_numpy(test_track[:, 1:, 1:])
    with torch.no_grad():
        means, covs = model(test_t)
        means = means[:, 1:, :]
        covs = covs[:, 1:, :]

        means = means.contiguous().view(means.size(0)*means.size(1), means.size(2))
        covs = covs.contiguous().view(covs.size(0)*covs.size(1), covs.size(2))
        target_t = target_t.contiguous().view(target_t.size(0)*target_t.size(1),
                                              target_t.size(2))

        res = means - target_t
        return res, covs
