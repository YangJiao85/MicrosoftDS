## Construct model
## Label: rate_spread
## 
import pandas as pd
import numpy as np
import numpy.random as nr
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import sklearn.model_selection as ms
import sklearn.metrics as sklm
from sklearn.model_selection import cross_val_score
import seaborn as sns
import scipy.stats as ss
import math

topdir = '/Users/yang/Documents/projDATA/MPP-DataScience/C11_Capstone/'
# Import data
Features = np.array(pd.read_csv(topdir + 't3_FeatureSelection/tol-0.99/hmda_features_red.csv'))
Feature_names = np.array(pd.read_csv(topdir + 't3_FeatureSelection/tol-0.99/hmda_features_red.csv').columns)
Labels = np.array(pd.read_csv(topdir + 't3_FeatureSelection/hmda_labels.csv'))
Labels = Labels.reshape(Labels.shape[0],)
print('Load Features and Labels')
print(Features.shape)
print(Labels.shape)

## Evaluate the model
def print_metrics(y_true, y_predicted):
    ## First compute R^2 and the adjusted R^2
    r2 = sklm.r2_score(y_true, y_predicted)
    #r2_adj = r2 - (n_parameters - 1)/(y_true.shape[0] - n_parameters) * (1 - r2)

    ## Print the usual metrics and the R^2 values
    print('Mean Square Error      = ' + str(sklm.mean_squared_error(y_true, y_predicted)))
    print('Root Mean Square Error = ' + str(math.sqrt(sklm.mean_squared_error(y_true, y_predicted))))
    print('Mean Absolute Error    = ' + str(sklm.mean_absolute_error(y_true, y_predicted)))
    print('Median Absolute Error  = ' + str(sklm.median_absolute_error(y_true, y_predicted)))
    print('R^2                    = ' + str(r2))
    #print('Adjusted R^2           = ' + str(r2_adj))

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

def print_cv(scores):
    fold = [x + 1 for x in range(len(scores['train_r2']))]
    print('         R2            MSE           MAE')
    [print_format(f,x,y,z) for f,x,y,z in zip(fold, scores['train_r2'],
                                          scores['train_neg_mean_squared_error'],
                                          scores['train_neg_mean_absolute_error'])]
    print('-' * 40)
    print('MEAN          %4.3f          %4.3f          %4.3f' %
          (np.mean(scores['train_r2']), np.mean(scores['train_neg_mean_squared_error']), np.mean(scores['train_neg_mean_absolute_error'])))
    print('Sdt           %4.3f          %4.3f          %4.3f' %
          (np.std(scores['train_r2']), np.std(scores['train_neg_mean_squared_error']), np.std(scores['train_neg_mean_absolute_error'])))

    fold = [x + 1 for x in range(len(scores['test_r2']))]
    print('         R2            MSE           MAE')
    [print_format(f,x,y,z) for f,x,y,z in zip(fold, scores['test_r2'],
                                          scores['test_neg_mean_squared_error'],
                                          scores['test_neg_mean_absolute_error'])]
    print('-' * 40)
    print('MEAN          %4.3f          %4.3f          %4.3f' %
          (np.mean(scores['test_r2']), np.mean(scores['test_neg_mean_squared_error']), np.mean(scores['test_neg_mean_absolute_error'])))
    print('Sdt           %4.3f          %4.3f          %4.3f' %
          (np.std(scores['test_r2']), np.std(scores['test_neg_mean_squared_error']), np.std(scores['test_neg_mean_absolute_error'])))

"""
#####################################################################
## Optimize hyperparameters with nested cross validation
nr.seed(123)
inside = ms.KFold(n_splits = 10, shuffle = True)
nr.seed(321)
outside = ms.KFold(n_splits = 10, shuffle = True)

## Define the dictionary for the grid search and the model object to search on
#param_grid = {"max_features": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100], 
#              "min_samples_leaf": [3, 5, 10, 20]}
param_grid = {"max_features": [10, 60, 100, 130],
              "min_samples_leaf": [3, 5, 10, 20]}
## Define the random forest model
nr.seed(3456)
rf_clf = RandomForestRegressor(n_estimators = 10, criterion = "mse", n_jobs = -1) # 

## Perform the grid search over the parameters
nr.seed(4455)
rf_clf = ms.GridSearchCV(estimator = rf_clf, param_grid = param_grid,
                      cv = inside, # Use the inside folds
                      scoring = "r2",
                      return_train_score = True)

## Fit the cross validated grid search over the data
rf_clf.fit(Features, Labels)
## And print the best parameter value
print('Best estimator max_features, min_samples_leaf: ')
print(rf_clf.best_estimator_.max_features)
print(rf_clf.best_estimator_.min_samples_leaf)

## Dump the model
from joblib import dump
dump(rf_clf, 'RandomForest_CV.joblib')
#rf_clf = load('RandomForest_CV.joblib')

## Perform the outer cross validation of the model
nr.seed(498)
cv_estimate = ms.cross_val_score(rf_clf, Features, Labels, 
                                 cv = outside) # Use the outside folds
print('Mean performance metric = %4.3f' % np.mean(cv_estimate))
print('STD of the metric       = %4.3f' % np.std(cv_estimate))
print('Outcomes by cv fold')
for i, x in enumerate(cv_estimate):
    print('Fold %2d    %4.3f' % (i+1, x))

"""
#####################################################################
## Build and test a model using the estimated optimal hyperparameters
## Split the dataset
nr.seed(1115)
indx = range(Features.shape[0])
indx = ms.train_test_split(indx, test_size = 20000)
X_train = Features[indx[0],:]
y_train = np.ravel(Labels[indx[0]])
X_test = Features[indx[1],:]
y_test = np.ravel(Labels[indx[1]])
print(X_train.shape)
print(X_test.shape)

## Define a random forest model object using the estimated optimal model hyperparameters
##   and then fits the model to the training data
nr.seed(1115)
rf_mod = RandomForestRegressor( n_estimators = 10, criterion = "mse", n_jobs = -1,
                                max_features = 60, min_samples_leaf = 10 )
                                #max_features = rf_clf.best_estimator_.max_features,
                                #min_samples_leaf = rf_clf.best_estimator_.min_samples_leaf)
rf_mod.fit(X_train, y_train)
y_score = rf_mod.predict(X_test)
y_score = y_score.round(1)
print_metrics(y_test, y_score)
hist_resids(y_test, y_score)
resid_qq(y_test, y_score)
resid_plot(y_test, y_score)

## Feature importance
def plot_feature_importance(rf_mod):
    importance = rf_mod.feature_importances_
    print(Feature_names)
    print(importance)
    d = {'feature_name': Feature_names, 'importance': importance}
    feature_importance = pd.DataFrame(data=d)
    feature_importance_red = feature_importance.loc[feature_importance['importance'] > 0.01]
    print(feature_importance_red)
    plt.bar(range(feature_importance_red.shape[0]), feature_importance_red['importance'], tick_label = feature_importance_red['feature_name'])
    plt.xticks(rotation=90)
    plt.ylabel('Feature importance')
    plt.tight_layout()
    plt.show()

plot_feature_importance(rf_mod)

nr.seed(1115)
rf_mod = RandomForestRegressor( n_estimators = 10, criterion = "mse", n_jobs = -1,
                                max_features = 60, min_samples_leaf = 10 )
                                #max_features = rf_clf.best_estimator_.max_features,
                                #min_samples_leaf = rf_clf.best_estimator_.min_samples_leaf)
rf_mod.fit(Features, Labels)
## Dump the model
from joblib import dump
dump(rf_mod, 'RandomForest.joblib')


