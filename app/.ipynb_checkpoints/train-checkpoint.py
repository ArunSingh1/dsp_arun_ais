from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error
import numpy as np
import joblib


def training_data_prep(df):
    y=df['SalePrice']
    X=df.drop(['SalePrice'],axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)
    return X_train, X_test, y_train, y_test

def model_train(classifier, X_train, y_train, X_test):
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    return y_pred


print('code working')


def initialise_models():
    linreg = LinearRegression()
    l2 = Ridge(alpha=1.0, normalize=True)
    l1 = Lasso(alpha=1.0, normalize=True)
    decision = DecisionTreeRegressor(random_state=0)
    regressor = RandomForestRegressor(random_state=0)
    return linreg, l2, l1, decision, regressor


def model_evalution(y_test, y_pred, classifier):
    MAE = np.floor(mean_absolute_error(y_test, y_pred))
    MSE = np.floor(mean_squared_error(y_test, y_pred))
    RMSE = np.floor(np.sqrt(mean_squared_error(y_test, y_pred)))
    rmsle = np.round(compute_rmsle(y_test, y_pred),3)
    print("Log Root Mean Squared Error", rmsle)
    
    #results_df = to_result_df(classifier, rmsle, MAE, MSE, RMSE, results_df)
    
    return 



def compute_rmsle(y_test: np.ndarray, y_pred: np.ndarray, precision: int = 2) -> float:
    rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
    return rmsle

def saving_models(classifier):
    #filename =  '/home/arun/Documents/dsp_arun_ais/models/basicpipeline/{fname}.plk'.format(fname = classifier)
    filename =  '/home/arun/Documents/dsp_arun_ais/models/mainpipeline/{fname}.plk'.format(fname = classifier)
    print(filename)
    joblib.dump(classifier, filename)
    return
