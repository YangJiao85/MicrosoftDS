## Construct model
## Label: rate_spread
## 
import pandas as pd
import sklearn.model_selection as ms
from sklearn import linear_model
import sklearn.metrics as sklm
from sklearn.model_selection import cross_val_score
import numpy as np
import numpy.random as nr
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
import math

# Import data
Features = np.array(pd.read_csv('hmda_features_prep.csv'))
Labels = np.array(pd.read_csv('hmda_labels_prep.csv'))
print('Load Features and Labels')
print(Features.shape)
print(Labels.shape)

## Split the dataset
nr.seed(1234)
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
lin_mod = linear_model.Ridge()
lin_mod.fit(X_train, y_train)

print(lin_mod.intercept_)
print(lin_mod.coef_)

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

y_score = lin_mod.predict(X_test)
y_score = y_score.round(1)
print_metrics(y_test, y_score)

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

## Simple cross validation model
Labels = Labels.reshape(Labels.shape[0],)
scoring = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'] 
lin_mod = linear_model.Ridge(alpha = 7.0)
print(sorted(sklm.SCORERS.keys()))
#['accuracy', 'adjusted_mutual_info_score', 'adjusted_rand_score', 'average_precision', 'balanced_accuracy', 'brier_score_loss', 'completeness_score', 'explained_variance', 'f1', 'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted', 'fowlkes_mallows_score', 'homogeneity_score', 'jaccard', 'jaccard_macro', 'jaccard_micro', 'jaccard_samples', 'jaccard_weighted', 'max_error', 'mutual_info_score', 'neg_log_loss', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_mean_squared_log_error', 'neg_median_absolute_error', 'normalized_mutual_info_score', 'precision', 'precision_macro', 'precision_micro', 'precision_samples', 'precision_weighted', 'r2', 'recall', 'recall_macro', 'recall_micro', 'recall_samples', 'recall_weighted', 'roc_auc', 'v_measure_score']
scores = ms.cross_validate(lin_mod, Features, Labels, scoring = scoring, cv = 10, return_train_score=True)

def print_format(f, x, y, z):
    print('Fold %2d    %4.3f        %4.3f      %4.3f' % (f, x, y, z))

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

print_cv(scores)

#####################################################################
## Optimize hyperparameters with nested cross validation
nr.seed(123)
inside = ms.KFold(n_splits = 10, shuffle = True)
nr.seed(321)
outside = ms.KFold(n_splits = 10, shuffle = True)

nr.seed(3456)
## Define the dictionary for the grid search and the model object to search on
param_grid = {"alpha": [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]}
## Define the regression model
lin_mod = linear_model.Ridge()

## Perform the grid search over the parameters
##   clf: cross validated grid search object
clf = ms.GridSearchCV(estimator = lin_mod, param_grid = param_grid,
                      cv = inside, # Use the inside folds
                      scoring = 'r2',
                      return_train_score = True)

## Fit the cross validated grid search over the data
clf.fit(Features, Labels)
keys = list(clf.cv_results_.keys())
print('keys ', keys)
print('keys[6:20] ', keys[6:20])
for key in keys[6:20]:
    print(clf.cv_results_[key])
## And print the best parameter value
print('Best estimator alpha: ')
print(clf.best_estimator_.alpha)

def plot_cv(clf, params_grid, param = 'alpha'):
    params = [x for x in params_grid[param]]

    keys = list(clf.cv_results_.keys())
    grid = np.array([clf.cv_results_[key] for key in keys[6:16]])
    means = np.mean(grid, axis = 0)
    stds = np.std(grid, axis = 0)
    print('Performance metrics by parameter')
    print('Parameter    Mean performance    STD performance')
    for x,y,z in zip(params, means, stds):
        print('%8.2f        %6.5f        %6.5f' % (x,y,z))

    #params = [math.log10(x) for x in params]

    plt.scatter(params * grid.shape[0], grid.flatten())
    p = plt.scatter(params, means, color = 'red', marker = '+', s = 300)
    plt.plot(params, np.transpose(grid))
    plt.title('Performance metric vs. parameter value\n from cross validation')
    plt.xlabel('Hyperparameter value')
    plt.ylabel('Performance metric')
    plt.show()

plot_cv(clf, param_grid)

#####################################################################
## Build and test a model using the estimated optimal hyperparameters
lin_mod = linear_model.Ridge(clf.best_estimator_.alpha)
lin_mod.fit(X_train, y_train)
y_score = lin_mod.predict(X_test)
y_score = y_score.round(1)
print_metrics(y_test, y_score)
hist_resids(y_test, y_score)
resid_qq(y_test, y_score)
resid_plot(y_test, y_score)


lin_mod = linear_model.Ridge(clf.best_estimator_.alpha)
lin_mod.fit(Features, Labels)
## Dump the model
from joblib import dump
dump(lin_mod, 'LinearModel_l2_cv.joblib')


