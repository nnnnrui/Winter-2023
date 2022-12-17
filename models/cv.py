from sklearn.multioutput import RegressorChain
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import yaml
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import RegressorChain
from sklearn.model_selection import cross_validate, cross_val_predict, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import csv
import itertools
import sys

def prepare_full_data(dataset = "dravnieks"):
    dataset = dataset
    data = pd.read_csv(f"data/{dataset}/raw/processed_descriptors.csv")
    target = pd.read_csv(f"data/{dataset}/raw/{dataset}.csv")
    target.dropna(inplace=True)
    ind_X = (data.columns=="CanonicalSMILES").argmax()+1
    ind_y = (target.columns=="IUPACName").argmax()+1

    df = pd.DataFrame(columns=target.columns)
    if data.shape[0]!=target.shape[0]:
        for cid in data["CID"]:
            if cid in target["CID"].values:
                df = pd.concat([df,target[target["CID"].values==cid]],axis=0)

    X_train = data.iloc[:,ind_X:].fillna(0)
    y_train = df.iloc[:,ind_y:]
    label = target.iloc[:,ind_y:].columns

    return (X_train,y_train,label)

if __name__=="__main__":
    dataset = "keller"
    data = pd.read_csv(f"../data/{dataset}/raw/processed_descriptors.csv")
    target = pd.read_csv(f"../data/{dataset}/raw/{dataset}.csv")
    target.dropna(inplace=True)
    ind_X = (data.columns=="CanonicalSMILES").argmax()+1
    ind_y = (target.columns=="IUPACName").argmax()+1

    df = pd.DataFrame(columns=target.columns)
    print(target.shape)
    if data.shape[0]!=target.shape[0]:
        for cid in data["CID"]:
            if cid in target["CID"].values:
                df = pd.concat([df,target[target["CID"].values==cid]],axis=0)

    X = data.iloc[:,ind_X:].fillna(0)
    Y = df.iloc[:,ind_y:].to_numpy()
    label = target.iloc[:,ind_y:].columns.to_numpy()

    # X_train, X_test, y_train, y_test=train_test_split(X,Y,test_size = 0.20, random_state=42)

    ## prepare model
    models = ["Ridge", "SVR", "RandomForestRegressor", "GradientBoostingRegressor"]
    multi_models=["reg_chain_Ridge","reg_chain_SVR","multi_output_RandomForestRegressor", "reg_chain_GradientBoostingRegressor"]

    mse_error = {}
    r2_error = {}
    corr_coef = {}
    for model_name in models:
        print(f"start cross-validating the model {model_name}")

        with open("../configs/best_param/"+model_name.lower()+".yaml", 'r') as stream:
            parameters = yaml.safe_load(stream)

        ## specify models with parameters
        if model_name=="Ridge":
            model=Ridge(**parameters)
        elif model_name=="SVR":
            model=SVR(**parameters)
        elif model_name=="RandomForestRegressor":
            model=RandomForestRegressor(**parameters)
        elif model_name=="GradientBoostingRegressor":
            model=GradientBoostingRegressor(**parameters)
        else:
            break
    
        ## cross-validation
        cv_error=dict()
        test_error_mse = []
        test_error_r2=[]
        test_corr_coef=[]
        for i in tqdm(np.arange(Y.shape[1])):
            lab = label[i]
            
            cv_pred = cross_val_predict(model, X,Y[:,i],cv=KFold(n_splits=3, shuffle=True, random_state=12))
            test_error_mse+=[mean_squared_error(Y[:,i],cv_pred)]
            test_error_r2+=[r2_score(Y[:,i],cv_pred)]
            test_corr_coef+=[pearsonr(Y[:,i],cv_pred).statistic]
        
        mse_error[model_name]=test_error_mse
        r2_error[model_name]=test_error_r2
        corr_coef[model_name] = test_corr_coef
        print(f"Done with {model_name}")

    df=pd.DataFrame()
    for model_name in multi_models:
        print(f"start cross-validating the model {model_name}")

        with open("../configs/best_param/"+model_name.lower()+"_param.yaml", 'r') as stream:
            parameters = yaml.safe_load(stream)
    
        ## cross-validation
        cv_error=dict()
        if model_name=="multi_output_RandomForestRegressor":
            model=RandomForestRegressor(**parameters)
        else:
            continue

        cv_pred = cross_val_predict(model, X,Y,cv=KFold(n_splits=3, shuffle=True, random_state=12))

        mse_error[model_name]=mean_squared_error(Y,cv_pred,multioutput="raw_values")
        r2_error[model_name]=r2_score(Y,cv_pred,multioutput="raw_values")
        print(f"Done with {model_name}")

    df_mse = pd.DataFrame(mse_error)
    df_r2 = pd.DataFrame(r2_error)

    df_mse.set_index(label, inplace=True)
    df_r2.set_index(label, inplace=True)

    # df_mse.to_csv(f"../outputs/within_dataset/cv_mse_{dataset}.csv")
    # df_r2.to_csv(f"../outputs/within_dataset/cv_r2_{dataset}.csv")