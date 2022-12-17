from datamodule.datasets.kellerdataset import KellerDataset
import pandas as pd
from models.within_dataset import prepare_full_data

import numpy as np

keller_feat = np.load(f"configs/best_param/lasso_feature_keller.npy")
drav_feat = np.load(f"configs/best_param/lasso_feature_dravnieks.npy")
keller,yk,labelk = prepare_full_data(dataset="keller")
dravnieks,yd,labeld = prepare_full_data(dataset="dravnieks")

select = np.sum(drav_feat,axis=0)
lasso_feat = dravnieks.columns[select>0].to_numpy()
with open(f'configs/best_param/lasso_feature.npy', 'wb') as f:
        np.save(f,lasso_feat)