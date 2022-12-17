from sklearn import metrics
from tqdm import tqdm
import pandas as pd
import numpy as np
import yaml
from sklearn.linear_model import Ridge,Lasso
from sklearn.svm import SVR
from sklearn.multioutput import RegressorChain
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings

from classic_ml import  log
from within_dataset import prepare_full_data


if __name__=="__main__":
    lasso_feat = np.load(f"configs/best_param/lasso_feature.npy",allow_pickle=True)
    share = pd.read_pickle("data/shared_labels.pkl")

    X_train, y_train, _ = prepare_full_data(dataset="dravnieks")
    # print(y_train)
    X_train = X_train[lasso_feat].to_numpy()
    label_train = share["Dravnieks"]
    y_train = y_train[label_train].to_numpy()

    X_test, y_test, _ = prepare_full_data(dataset="keller")
    X_test = X_test[lasso_feat].to_numpy()
    label_test = share["Keller"]
    y_test = y_test[label_test]

    ## prepare model
    models = ["Lasso", "SVR", "RandomForestRegressor", "GradientBoostingRegressor"]
    multi_models=["reg_chain_Lasso","reg_chain_SVR","multi_output_RandomForestRegressor", "reg_chain_GradientBoostingRegressor"]
    keys = label_test+" vs. "+label_train

    mse_error=dict(zip(multi_models+models, [None]*len(multi_models+models)))
    r2_error=dict(zip(multi_models+models, [None]*len(multi_models+models)))
    for model_name in models:

        ## train-test 
        if model_name!="Lasso":
            with open("configs/best_param/"+model_name.lower()+"_param.yaml", 'r') as stream:
                parameters = yaml.safe_load(stream)

        test_error_mse=[]
        test_error_r2=[]
        for i in tqdm(np.arange(y_train.shape[1])):
            label = label_train[i]
            test_label = label_test[i]

            ## specify models with parameters
            if "Lasso" in model_name:
                model=Lasso()
            elif "SVR" in model_name:
                model=SVR(**parameters[label])
            elif "RandomForest" in model_name:
                model=RandomForestRegressor(**parameters[label])
            elif "GradientBoosting" in model_name:
                model=GradientBoostingRegressor(**parameters[label])
            else:
                break
            
            result = model.fit(X_train,y_train[:,i])
            preds = model.predict(X_test)

            # print(y_test[test_label])
            test_error_mse+=[mean_squared_error(y_pred=preds, y_true=y_test[test_label])]
            test_error_r2+=[r2_score(y_pred=preds, y_true=y_test[test_label])]

        mse_error[model_name]=test_error_mse
        r2_error[model_name]=test_error_r2
    
    ## multi-output model

    for model_name in tqdm(multi_models):
        ## load parameters
        if "Lasso" not in model_name:
            with open("configs/best_param/"+model_name.lower()+"_param.yaml", 'r') as stream:
                parameters = yaml.safe_load(stream)

        if "RandomForestRegressor" in model_name:
            model=RandomForestRegressor(**parameters)
        elif "GradientBoostingRegressor" in model_name:
            ml=GradientBoostingRegressor()
            model = RegressorChain(ml, order=list(parameters["order"]), random_state=parameters["random_state"])
        elif "SVR" in model_name:
            ml=SVR()
            model = RegressorChain(ml, order=list(parameters["order"]), random_state=parameters["random_state"])
        elif "Lasso" in model_name:
            ml=Lasso()
            model = RegressorChain(ml, random_state=42)

        result = model.fit(X_train,y_train)
        preds = model.predict(X_test)
        mse_error[model_name]=mean_squared_error(y_pred=preds, y_true=y_test, multioutput="raw_values")
        r2_error[model_name]=r2_score(y_pred=preds, y_true=y_test, multioutput="raw_values")
    
    df_mse = pd.DataFrame(mse_error)
    df_r2 = pd.DataFrame(r2_error)

    df_mse.set_index(keys, inplace=True)
    df_r2.set_index(keys,inplace=True)

    df_mse.to_csv("outputs/lasso_reduct/mse_ml.csv")
    df_r2.to_csv("outputs/lasso_reduct/r2_ml.csv")