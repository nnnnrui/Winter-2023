import numpy as np
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
import os
# from classic_ml import prepare_data
from within_dataset import prepare_full_data
import pandas as pd

dataset = "dravnieks"
X_train,y_train,label = prepare_full_data(dataset=dataset)

lasso_feat = np.load(f"configs/best_param/lasso_feature.npy",allow_pickle=True)

models = ["Lasso", "SVR", "RandomForestRegressor", "GradientBoostingRegressor"]

mse = dict(); r2 = dict()
for model_name in models:
    print(f"start cross-validating the model {model_name}")

    if model_name=="Lasso":
        model=Lasso(alpha=5)
    elif model_name=="SVR":
        model=SVR()
    elif model_name=="RandomForestRegressor":
        model=RandomForestRegressor()
    elif model_name=="GradientBoostingRegressor":
        model=GradientBoostingRegressor()
    else:
        break
    
    mse_error = dict()
    r2_error = dict()
    for i in tqdm(np.arange(len(label))):
        X = X_train[lasso_feat].to_numpy()
        cv_pred = cross_val_predict(model, X,y_train[:,i],cv=KFold(n_splits=3, shuffle=True, random_state=12))

        mse_error[label[i]]=mean_squared_error(y_train[:,i],cv_pred)
        r2_error[label[i]]=r2_score(y_train[:,i],cv_pred)
    
    mse[model_name] = mse_error
    r2[model_name] = r2_error
    print(f"Done with {model_name}")

pd.DataFrame(mse).to_csv(f"outputs/lasso_reduct/mse_pair_{dataset}.csv")
pd.DataFrame(r2).to_csv(f"outputs/lasso_reduct/r2_pair_{dataset}.csv")
    
    