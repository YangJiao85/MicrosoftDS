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
from joblib import load

f_model = 'LinearModel_fs_l2_cv.joblib'
f_pred = 'hmda_pred_fs_l2_cv.csv'
## Load the model
lin_mod = load(f_model)
# Import data
X_new = np.array(pd.read_csv('hmda_features_new_red.csv'))

y_predict = lin_mod.predict(X_new).round(1)
print(type(y_predict))
print(len(y_predict))

## Print predicted labels 
hmda_test = pd.read_csv('../test_values.csv')
hmda_pred = pd.DataFrame({'row_id': np.array(hmda_test['row_id']), \
                          'rate_spread': y_predict[:]})
print('Written predictions to file ' + f_pred)
hmda_pred.to_csv(f_pred, index=False)

