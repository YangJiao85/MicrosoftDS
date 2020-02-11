## Construct model
## Labels
## Category: BikeBuyer
## Numeric:  AveMonthSpend --> log_AveMonthSpend
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

# Import data for customer information
aw = pd.read_csv('AW_prep.csv')
print( 'Load AW_prep.csv')

cat_cols = ['Education', 'Occupation', 'Gender', 'MaritalStatus', 'HomeOwnerFlag', 'NCarsGroup', 'NChildrenAtHomeGroup', 'TotalChildrenGroup']
num_cols = ['YearlyIncome', 'Age']

## Prepare the model matrix
## Categorical features

def encode_string(cat_feature):
    ## First encode the strings to numeric categories
    enc = preprocessing.LabelEncoder()
    enc.fit(cat_feature)
    enc_cat_feature = enc.transform(cat_feature)
    ## Now, apply one hot encoding
    ohe = preprocessing.OneHotEncoder(categories='auto')
    encoded = ohe.fit(enc_cat_feature.reshape(-1,1))
    return encoded.transform(enc_cat_feature.reshape(-1,1)).toarray()

for col in cat_cols:
    temp = encode_string(aw[col])
    if (col == cat_cols[0]):
        Features = temp
    else:
        Features = np.concatenate([Features, temp], axis = 1)

print('Transformed categorical features')
print(Features.shape)
print(Features[:2, :])

## Numeric features
Features = np.concatenate([Features, np.array(aw[num_cols])], axis = 1)
print('Transformed features')
print(Features.shape)
print(Features[:2,:])

## Split the dataset
nr.seed(1234)
labels = np.array(aw['log_AveMonthSpend'])
indx = range(Features.shape[0])
indx = ms.train_test_split(indx, test_size = 2000)
X_train = Features[indx[0],:]
y_train = np.ravel(labels[indx[0]])
X_test = Features[indx[1],:]
y_test = np.ravel(labels[indx[1]])

## Rescale numeric features
scaler = preprocessing.StandardScaler().fit(X_train[:,27:])
X_train[:,27:] = scaler.transform(X_train[:,27:])
X_test[:,27:] = scaler.transform(X_test[:,27:])
print(X_train.shape)
print(X_train[:5,:])

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
print_metrics(y_test, y_score, 39)

##   Desplay the residuals  
def hist_resids(y_test, y_score):
    ## first compute vector of residuals. 
    resids = np.subtract(y_test.reshape(-1,1), y_score.reshape(-1,1))
    ## now make the residual plots
    sns.distplot(resids)
    plt.title('Histogram of residuals')
    plt.xlabel('Residual value')
    plt.ylabel('count')
    plt.show()

hist_resids(y_test, y_score)

######################################################################
## Predict new input features
aw_cust_test = pd.read_csv('AW_test.csv')
print(aw_cust_test.columns)
print(aw_cust_test.shape)

def trans_features(aw_custs):
    ## new column 'age' by data collected date 1st January 1998 - 'BirthDate'
    aw_custs['Age'] = (pd.to_datetime('1998-01-01') - pd.to_datetime(aw_custs['BirthDate'], errors='coerce')).astype('<m8[Y]')
    ##    Aggregating categorical variables
    ## AgeGroup <25 , 25-45, 45-55, >55
    aw_custs['AgeGroup'] = pd.cut(aw_custs['Age'], bins = [0,25,45,55,1000], \
              labels = ['<25','25-45','45-55','>55'], right=False)
    aw_custs['NCarsGroup'] = pd.cut(aw_custs['NumberCarsOwned'], \
              bins = [-1,0,2,10], labels = ['No','1-2','>=3'], right=True)
    aw_custs['NChildrenAtHomeGroup'] = pd.cut(aw_custs['NumberChildrenAtHome'], \
              bins = [-1,0,100], labels = ['No','>=1'], right=True)
    aw_custs['TotalChildrenGroup'] = pd.cut(aw_custs['TotalChildren'], \
              bins = [-1,0,1,2,3,4,100], labels = ['No','1','2','3','4','>=5'], right=True)

    for col in cat_cols:
        temp = encode_string(aw_custs[col])
        if (col == cat_cols[0]):
            Features = temp
        else:
            Features = np.concatenate([Features, temp], axis = 1)

    print('Transformed categorical features')
    print(Features.shape)
    print(Features[:2, :])

    ## Numeric features
    Features = np.concatenate([Features, np.array(aw_custs[num_cols])], axis = 1)
    Features[:,27:] = scaler.transform(Features[:,27:])
    print('Transformed features')
    print(Features.shape)
    print(Features[:2,:])
    return Features

X_new = trans_features(aw_cust_test)
y_predict = lin_mod.predict(X_new)
print(type(y_predict))
print(len(y_predict))
print(type(np.array(aw_cust_test['CustomerID'])))
print(len(np.array(aw_cust_test['CustomerID'])))

y_predict_untransform = np.exp(y_predict)
aw_AveMonthSpend_pred = pd.DataFrame({'CustomerID': np.array(aw_cust_test['CustomerID']),'AveMonthSpend_pred': y_predict_untransform[:]})
aw_AveMonthSpend_pred.to_csv('AW_AveMonthSpend_pred.csv',index=False)

hist_resids(y_predict, y_score[:500])

