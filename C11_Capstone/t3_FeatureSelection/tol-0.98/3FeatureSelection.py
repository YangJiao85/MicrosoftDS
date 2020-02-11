import pandas as pd
from sklearn import preprocessing
import sklearn.model_selection as ms
from sklearn import linear_model
import sklearn.metrics as sklm
from sklearn import feature_selection as fs
from sklearn import metrics
import numpy as np
import numpy.random as nr
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
import math
import sys
np.set_printoptions(threshold=sys.maxsize)

ddir = '../'
Feature_names = np.array(pd.read_csv(ddir + 'hmda_features.csv').columns)
Features = np.array(pd.read_csv(ddir + 'hmda_features.csv'))
Labels = np.array(pd.read_csv(ddir + 'hmda_labels.csv'))
print('Shape of Fetures and Labels')
print(Features.shape)
print(Labels.shape)
print(Feature_names)

## Eliminate low variance features
##   using a p = 0.95 threshold
## Define the variance threshold and fit the threshold to the feature array.
sel = fs.VarianceThreshold(threshold = (.98 * (1. - .98)))
Features_reduced = sel.fit_transform(Features)

## Print the support and shape for the transformed features
print(sel.get_support())
print(Features_reduced.shape)
Feature_names_red = Feature_names[sel.get_support()]
print(Feature_names_red)

## Feature selection for test set
Features_new = np.array(pd.read_csv(ddir + 'hmda_features_new.csv'))
print('Shape of Fetures for new dataset')
print(Features_new.shape)
Features_new_reduced = Features_new[:, sel.get_support()]
print(Features_new_reduced.shape)

## Select k best features
## Reshape the Label array
Labels = Labels.reshape(Labels.shape[0],)

## Set folds for nested cross validation
nr.seed(988)
feature_folds = ms.KFold(n_splits=10, shuffle = True)

## Define the model
lin_mod = linear_model.Ridge()

## Perform feature selection by CV with high variance features only
nr.seed(6677)
selector = fs.RFECV(estimator = lin_mod, cv = feature_folds, scoring = 'r2')
selector = selector.fit(Features_reduced, Labels)
print(selector.support_)
print(selector.ranking_)
print(Feature_names_red[selector.get_support()])

## Apply the selector to the feature array
Features_reduced = selector.transform(Features_reduced)
print(Features_reduced.shape)

## Feature selection for test set
print('Shape of Fetures for new dataset')
Features_new_reduced = Features_new_reduced[:, selector.get_support()]
print(Features_new_reduced.shape)

## Save selected features
hmda_features_sel = pd.DataFrame(Features_reduced)
hmda_features_sel.to_csv('hmda_features_red.csv', index=False)
hmda_features_new_sel = pd.DataFrame(Features_new_reduced)
hmda_features_new_sel.to_csv('hmda_features_new_red.csv', index=False)

## r2 plot vs. the number of features
def plot_r2_nf(selector):
    plt.plot(range(1, len(selector.grid_scores_) + 1), selector.grid_scores_)
    plt.title(r'Mean $R^2$ by number of features')
    plt.ylabel(r'$R^2$')
    plt.xlabel('Number of features')
    plt.show()

plot_r2_nf(selector)


