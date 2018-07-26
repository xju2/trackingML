"""
For each track, assuming there are 20 hits, each hits having 3 coordinates.
So each track has a input dimension of 60.
If less than 20 hits, fill with a large value.

Output: probablity of the input hits to be a real track.
"""

import torch
import torch.nn as nn

import sys
sys.path.insert(0, '../common')

from utils import device


class IsGoodTrack(nn.Module):

    def __init__(self, input_dim=60, hidden_dim=120, batch_size=64, n_lstm_layers=1, device=None):

        super(IsGoodTrack, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_lstm_layers, batch_first=True, dropout=0.1)
        self.fc1 = nn.Linear(hidden_dim, int(input_dim/2))
        self.fc2 = nn.Linear(int(input_dim/2), 1)
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
        output = self.fc2(output)
        output = self.sigma(output)
        return output
