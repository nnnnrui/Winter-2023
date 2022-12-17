# from classic_ml import prepare_data
from within_dataset import prepare_full_data
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from tqdm import tqdm
import numpy as np
import pandas as pd



if __name__=="__main__":
    dataset = "keller"

    X_train,y_train,label = prepare_full_data(dataset)

    X_train_train, X_train_test,y_train_train,y_train_test = train_test_split(X_train, y_train, test_size=0.33, random_state=42)

    ## param search
    param = np.logspace(-2,2,num=5)
    error = np.zeros(len(param))
    i = 0
    for alpha in tqdm(param):
        lasso = Lasso(alpha=alpha, random_state=42)
        lasso.fit(X_train_train,y_train_train)
        param_pred = lasso.predict(X_train_test)
        error[i] = r2_score(y_train_test,param_pred)
        i+=1
        # print(error[i])
    
    ind = np.argmax(error)
    print(f"best r2 score is {np.max(error)}, achieved by the alpha {param[ind]}")

    lasso = Lasso(alpha=param[ind], random_state=42)
    lasso.fit(X_train,y_train)
    print(lasso.coef_.shape)
    with open(f'configs/best_param/lasso_feature_{dataset}.npy', 'wb') as f:
        np.save(f,lasso.coef_>0)

