## Construct model
## Label: rate_spread
## 
import pandas as pd
from sklearn import preprocessing
import sklearn.model_selection as ms
from sklearn import linear_model
import sklearn.metrics as sklm
import numpy as np
import numpy.random as nr
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
import math

# Import data
Features = np.array(pd.read_csv('hmda_features_red.csv'))
Labels = np.array(pd.read_csv('hmda_labels.csv'))
print('Load Features and Labels')
print(Features.shape)
print(Labels.shape)

## Split the dataset
nr.seed(1234)
#labels = np.array(hmda_train['rate_spread'])
indx = range(Features.shape[0])
indx = ms.train_test_split(indx, test_size = 20000)
X_train = Features[indx[0],:]
y_train = np.ravel(Labels[indx[0]])
X_test = Features[indx[1],:]
y_test = np.ravel(Labels[indx[1]])
print(X_train.shape)
print(X_test.shape)

## Construct the linear regression model
##   Define and fit the linear regression model
lin_mod = linear_model.LinearRegression(fit_intercept = False)
lin_mod.fit(X_train, y_train)

print(lin_mod.intercept_)
print(lin_mod.coef_)

## Evaluate the model
def print_metrics(y_true, y_predicted, n_parameters):
    ## First compute R^2 and the adjusted R^2
    r2 = sklm.r2_score(y_true, y_predicted)
    r2_adj = r2 - (n_parameters - 1)/(y_true.shape[0] - n_parameters) * (1 - r2)

    ## Print the usual metrics and the R^2 values
    print('Mean Square Error      = ' + str(sklm.mean_squared_error(y_true, y_predicted)))
    print('Root Mean Square Error = ' + str(math.sqrt(sklm.mean_squared_error(y_true, y_predicted))))
    print('Mean Absolute Error    = ' + str(sklm.mean_absolute_error(y_true, y_predicted)))
    print('Median Absolute Error  = ' + str(sklm.median_absolute_error(y_true, y_predicted)))
    print('R^2                    = ' + str(r2))
    print('Adjusted R^2           = ' + str(r2_adj))

y_score = lin_mod.predict(X_test)
y_score = y_score.round(1)
print_metrics(y_test, y_score, 500)

##   Desplay the residuals  
def resid_plot(y_test, y_score):
    ## first compute vector of residuals.
    resids = np.subtract(y_test.reshape(-1,1), y_score.reshape(-1,1))
    ## now make the residual plots
    sns.regplot(y_score, resids, fit_reg=False)
    plt.title('Residuals vs. predicted values')
    plt.xlabel('Predicted values')
    plt.ylabel('Residual')
    plt.show()

def hist_resids(y_test, y_score):
    ## first compute vector of residuals. 
    resids = np.subtract(y_test.reshape(-1,1), y_score.reshape(-1,1))
    ## now make the residual plots
    sns.distplot(resids)
    plt.title('Histogram of residuals')
    plt.xlabel('Residual value')
    plt.ylabel('count')
    plt.show()

def resid_qq(y_test, y_score):
    ## first compute vector of residuals.
    resids = np.subtract(y_test, y_score)
    ## now make the residual plots
    ss.probplot(resids.flatten(), plot = plt)
    plt.title('Residuals vs. predicted values')
    plt.xlabel('Predicted values')
    plt.ylabel('Residual')
    plt.show()


hist_resids(y_test, y_score)
resid_qq(y_test, y_score)
resid_plot(y_test, y_score)

## Dump the model
from joblib import dump
dump(lin_mod, 'SimplestLinearModel.joblib')


