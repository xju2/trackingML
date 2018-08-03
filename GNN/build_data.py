"""
Build graph for each event
"""
import numpy as np
import pandas as pd
import glob

from trackml.dataset import load_event

import os
import sys
sys.path.insert(0, '../common')

from utils import get_features
from graph import construct_graph
from graph import save_graph

phi_slope_max = 0.001
z0_max_inner = 100
z0_max_outer = 50


def get_layer_pairs():
    start_layers = [8, 9, 10, 11, 25, 26, 27, 28, 41]
    end_layers = [9, 10, 11, 25, 26, 27, 28, 41, 42]
    layer_pairs = np.stack([start_layers, end_layers], axis=1)
    return layer_pairs

def graph_from_event(path, layer_pairs):
    hits, cells, particles, truth = load_event(path)
    hits = get_features(hits)
    hits_truth = hits.merge(truth, on='hit_id')
    hits_truth['barcode'] = hits_truth['particle_id']
    hits_truth['evtid'] = 0

    keys = ['evtid', 'barcode', 'r2',
            'eta', 'phi', 'z', 'volume_id', 'layer_id']
    hits4gg = pd.DataFrame(hits_truth[keys])
    layer_idx = pd.read_csv('layer_id.csv')

    hits4gg = hits4gg.merge(layer_idx, on=['volume_id', 'layer_id'])
    hits4gg['r'] = hits4gg['r2']
    hits_barrel = hits4gg[ (hits4gg['eta'] < 0.1) & (hits4gg['eta'] > -0.1) ]

    graph = construct_graph(hits_barrel, layer_pairs, phi_slope_max, z0_max_inner, z0_max_outer)
    return graph

def build_graphs(path, output_dir):
    train = np.unique([p.split('-')[0] for p in sorted(glob.glob(path + '/**'))])
    layer_pairs = get_layer_pairs()
    file_names = [os.path.join(output_dir, 'event%06i'%i) for i in range(len(train))]
    for event, file_name in zip(train, file_names):
        graph = graph_from_event(event, layer_pairs)
        save_graph(graph, file_name)
