#!/bin/bash
afix="determine"
#scp electra.lbl.gov:/home/xju/code/trackingML/model_hitGaus model_hitGaus_${afix}
#scp electra.lbl.gov:/home/xju/code/trackingML/ten_hists_normed.npy input/.
#scp electra.lbl.gov:/home/xju/code/trackingML/loss_trainGauss.pkl loss_trainGauss_${afix}.pkl

scp electra.lbl.gov:/home/xju/code/trackingML/output/data/train_data_uID.pkl.gz output/data/
afix="RNNModule_1"
scp electra.lbl.gov:/home/xju/code/trackingML/output/model/$afix output/model/
scp electra.lbl.gov:/home/xju/code/trackingML/output/loss/$afix output/loss/
