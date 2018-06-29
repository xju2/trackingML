#!coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def cal_res(test_track):
    """
    calculate predicted residual and variances
    of the model, for this test_track.
    test_track's size: [batch_size, n_hits, 3]"""

    test_t = torch.from_numpy(test_track)
    target_t = torch.from_numpy(test_track[:, 1:, 1:])
    with torch.no_grad():
        output = model(test_t)
        means = output[:, :, 0:2]
        covs = output[:, :, 2:4]

        means = means.contiguous().view(means.size(0)*means.size(1), means.size(2))
        covs = covs.contiguous().view(covs.size(0)*covs.size(1), covs.size(2))
        target_t = target_t.contiguous().view(target_t.size(0)*target_t.size(1),
                                        target_t.size(2))

        res = means - target_t
        return res, covs

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
    dx1 = targets[:, 0] - outputs[:, 0]
    dx2 = targets[:, 1] - outputs[:, 1]
    c1 =  outputs[:, 2]
    c2 =  outputs[:, 3]
    rho = outputs[:, 4]

    det_sigma = (1 - rho*rho) * c1 * c2
    log_det = torch.log(det_sigma)
    chi2 = (dx1*dx1/c1 + dx2*dx2/c2 - 2*rho*dx1*dx2/torch.sqrt(c1*c2))/(1-rho*rho)
    prob = log_det + chi2

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
        print(output[0])
        return output
