from sklearn import metrics
from tqdm import tqdm
import pandas as pd
import numpy as np
import yaml
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.multioutput import RegressorChain
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings

from classic_ml import prepare_data, log


if __name__=="__main__":
    train_wrap = prepare_data(dataset="Dravnieks",numpy_form=True)
    test_wrap = prepare_data(dataset="Keller",numpy_form=False, test=True)

    X_train = train_wrap["data"]
    y_train = train_wrap["target"]
    label_train=train_wrap["shared_labels"]

    X_test = test_wrap["data"]
    y_test = test_wrap["target"]
    label_test=test_wrap["shared_labels"]

    ## prepare model
    models = ["Ridge", "SVR", "RandomForestRegressor", "GradientBoostingRegressor"]
    multi_models=["reg_chain_Ridge","reg_chain_SVR","multi_output_RandomForestRegressor", "reg_chain_GradientBoostingRegressor"]
    keys = label_test+" vs. "+label_train

    mse_error=dict(zip(multi_models+models, [None]*len(multi_models+models)))
    r2_error=dict(zip(multi_models+models, [None]*len(multi_models+models)))
    for model_name in models:

        ## train-test 
        with open("configs/best_param/"+model_name.lower()+"_param.yaml", 'r') as stream:
            parameters = yaml.safe_load(stream)

        test_error_mse=[]
        test_error_r2=[]
        for i in tqdm(np.arange(y_train.shape[1])):
            label = label_train[i]
            test_label = label_test[i]

            ## specify models with parameters
            if "Ridge" in model_name:
                model=Ridge(**parameters[label])
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

            test_error_mse+=[mean_squared_error(y_pred=preds, y_true=y_test[test_label])]
            test_error_r2+=[r2_score(y_pred=preds, y_true=y_test[test_label])]

        mse_error[model_name]=test_error_mse
        r2_error[model_name]=test_error_r2
    
    ## multi-output model

    for model_name in tqdm(multi_models):
        ## load parameters
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
        elif "Ridge" in model_name:
            ml=Ridge()
            model = RegressorChain(ml, order=list(parameters["order"]), random_state=parameters["random_state"])

        result = model.fit(X_train,y_train)
        preds = model.predict(X_test)
        mse_error[model_name]=mean_squared_error(y_pred=preds, y_true=y_test, multioutput="raw_values")
        r2_error[model_name]=r2_score(y_pred=preds, y_true=y_test, multioutput="raw_values")
    
    df_mse = pd.DataFrame(mse_error)
    df_r2 = pd.DataFrame(r2_error)

    df_mse.set_index(keys, inplace=True)
    df_r2.set_index(keys,inplace=True)

    df_mse.to_csv("outputs/train_test/mse_classic_ml.csv")
    df_r2.to_csv("outputs/train_test/r2_classic_ml.csv")