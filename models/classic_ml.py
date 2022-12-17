from heapq import merge
import pandas as pd
import numpy as np
import logging
from sklearn.multioutput import RegressorChain
from random import sample
import yaml
import os
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import RegressorChain
from itertools import permutations
import warnings

warnings.filterwarnings("ignore")

def prepare_data(dataset="Dravnieks", numpy_form=True, expand_dim=True, test=False, hedonic=False):
    """[preprocess data to fit the model input format]
    
    Arguments:
        dataset {string} -- name of the dataset we are using: dravnieks or keller
        shared_labels {bool} -- whether to return selected labels for transfer learning
        expand_dim {bool} -- whether to expand dimensions on the target data, 
                    which can be informed by the input format of the model we plan to use
    Returns:
        [dict] -- {data set, target}
    """
    target = pd.read_csv(f"../data/{dataset}/raw/processed_{dataset}.csv")
    target.dropna(inplace=True)   

    share = pd.read_pickle("../data/shared_labels.pkl")[dataset]

    data= pd.read_csv(f"../data/{dataset}/raw/processed_descriptors.csv")
    ind = (data.columns=="CanonicalSMILES").argmax()+1
    
    if test:
        ovlp = np.load("../notebooks/overlap.npy")
        for cid in ovlp:
            data = data[data["CID"]!=cid]
            target = target[target["CID"]!=cid]

    X = data.iloc[:,ind:].fillna(0)

    if hedonic:
        if dataset=="Dravnieks":
            Y=target["HEDONICS"]
        else:
            Y=target["PLEASANTNESS"]
    else:
        Y = target[share.to_numpy()]

    if numpy_form:
        X = X.to_numpy()
        Y = Y.to_numpy()
    
    return {"data":X, "target":Y, "shared_labels":share}


def log(path, file):
    """[Create a log file to record the experiment's logs]
    
    Arguments:
        path {string} -- path to the directory
        file {string} -- file name
    
    Returns:
        [obj] -- [logger that record logs]
    """

    # check if the file exist
    log_file = os.path.join(path, file)

    if not os.path.isfile(log_file):
        open(log_file, "w+").close()

    console_logging_format = "%(levelname)s %(message)s"
    file_logging_format = "%(levelname)s: %(asctime)s: %(message)s"

    # configure logger
    logging.basicConfig(level=logging.INFO, format=console_logging_format)
    logger = logging.getLogger()
    
    # create a file handler for output file
    handler = logging.FileHandler(log_file)

    # set the logging level for log file
    handler.setLevel(logging.INFO)
    
    # create a logging format
    formatter = logging.Formatter(file_logging_format)
    handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)

    return logger


if __name__=='__main__':

    # model type
    ml_model = KNeighborsRegressor()
    # model_name = "reg_chain_"+str(ml_model)[:len(str(ml_model))-2]
    model_name=str(ml_model)[:len(str(ml_model))-2]

    # prepare data for grid search
    dataset_name="Dravnieks"

    data_wrap = prepare_data(dataset=dataset_name, expand_dim=False, hedonic=False)

    X=data_wrap["data"]
    y=data_wrap["target"]
    shared_labels = data_wrap["shared_labels"]

    logger = log(path="../logs/", file=model_name.lower()+".logs")
    logger.info("-"*15+"Start Session!"+"-"*15)

    # load grid parameters
    with open("../configs/param_search/"+model_name.lower()+".yaml", 'r') as stream:
        parameters = yaml.safe_load(stream)

    logger.info("{} regressor parameter grid search".format(model_name))

    # start grid search
    # univariate
    grid_search = GridSearchCV(ml_model, parameters,
        cv=KFold(n_splits=3, shuffle=True, random_state=12), scoring="neg_mean_squared_error")
    grid_search.fit(X, y)
    best_param = grid_search.best_params_
    # for i in np.arange(y.shape[1]): 
    #     grid_search = GridSearchCV(ml_model, parameters,
    #     cv=KFold(n_splits=3, shuffle=True, random_state=12), scoring="neg_mean_squared_error")
    #     

    #     best_param[shared_labels.iloc[i]] = grid_search.best_params_
    #     logger.info("Best parameter for the label {} is{}".format(shared_labels.iloc[i],grid_search.best_params_))

    # regression chain

    # ml_model=Ridge(solver="saga")
    # order = sample(list(permutations(np.arange(y.shape[1]))),100)
    # parameters_={"order": order, "random_state": [12]}
    # wrapper = RegressorChain(ml_model)
    # print(wrapper)
    logger.info("Best parameter is{}".format(grid_search.best_params_))

    with open(f'../configs/best_param/{model_name.lower()}_param.yaml', 'w+') as outfile:
        yaml.dump(best_param, outfile)
    
