#!/usr/bin/env python

from process_data import training_data_with_eta_cut

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device: ", device)

from utils import make_uID

# prepare the data.
eta_cut = 3.5
hits, cells, particles, truth = load_event(os.path.join(train_dir, event_prefix))
hits = get_features(hits)
high_eta_hits = hits[(hits['eta'] > eta_cut)]
uID_for_higheta = make_uID(high_eta_hits)
high_eta_hits_uID = pd.merge(high_eta_hits, uID_for_higheta, on=['volume_id', 'layer_id', 'module_id'])
train_data = high_eta_hits_uID.merge(truth, on='hit_id')[['uID', 'particle_id']]

modules = uID_for_higheta.shape[0]

