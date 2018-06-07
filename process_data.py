"""
Scripts to prepare the dataset for training and testing
"""

from trackml.dataset import load_event
from trackml.randomize import shuffle_hits
from trackml.score import score_event

import torch

import pandas as pd

import glob
import os
import logging

# import from load modules
from utils import make_uID
from utils import random_choice

def filter_truth(truth):
    return truth[ (truth['weight'] > 5e-7 ) & (truth['particle_id'] != 0) ]

class data_uID(object):
    """
    Each hit is represented by a unique ID,
    derived from its associated volume ID, layer ID and module ID
    """
    def __init__(self, input_path='input'):
        self.det = pd.read_csv(os.path.join(input_path, 'detectors.csv'))
        self.det_uID = make_uID(self.det)
        self.modules = self.det_uID.shape[0]

    def load_training(path='input/train_1/event000001000'):
        if hasattr(self, 'data_pool'):
            for train, truth in data_pool:
                del train
                del truth
            del self.data_pool

        self.data_pool = []
        read = self.read_event
        if "event" in path:
            self.data_pool.append(read(path))
        else:
            train = np.unique([p.split('-')[0] for p in sorted(glob.glob(os.path.join(path, '**')))])
            for event in train:
                self.data_pool.append(read(event))

        logging.info('total events:', len(self.data_pool))
        logging.info('total truth particles:', sum([y.shape[0] for x,y in self.data_pool]))

    def read_event(self, path):
        hits, truth = load_event(path, parts=['hits', 'truth'])
        hits_with_uID = pd.merge(hits, self.det_uID, on=['volume_id', 'layer_id', 'module_id'])
        training = hits_with_uID.merge(filter_truth(truth), on='hit_id')[['uID', 'particle_id']]
        unique_truth = pd.Series(np.unique(training['particle_id']))
        return training, unique_truth

    def get_a_track(self):
        """ randomly pick a particle,
        return it's associated list of hits"""
        if not hasattr(self, 'data_pool'):
            self.load_training()

        train, truth = random_choice(self.data_pool)
        pID = truth.sample(1).values
        hits = train[train['particle_id'] == pID[0]]['uID'].values
        return hits

    def random_input_target(self):
        """
        randomly chose one track from track pool
        """
        hits = self.get_a_track()
        input_ = self.input_hits(hits)
        target_ = self.target_hits(hits)
        return input_, target_

    def input_hits(self, series):
        tensor = torch.zeros(len(series), 1, self.modules+1)
        for idx, h in enumerate(series):
            tensor[idx][0][h] = 1
        return tensor

    def target_hits(self, series):
            module_idx = series[1:].tolist()
            module_idx.append(self.modules)
            return torch.LongTensor(module_idx)
