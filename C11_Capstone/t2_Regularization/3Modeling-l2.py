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
from DataJoin import clean_hmda_data

# Import data
hmda_train = pd.read_csv('hmda_train.csv')
print('Load hmda_train.csv')
print(hmda_train.shape)

#cat_cols = ['msa_md', 'state_code', 'county_code', \
#            'lender', 'loan_type', 'property_type', 'loan_purpose', 'occupancy', 'preapproval', \
#            'applicant_ethnicity', 'applicant_race', 'applicant_sex', 'co_applicant'] 
num_cols = ['loan_amount', 'applicant_income', \
            'population', 'minority_population_pct', 'ffiecmedian_family_income', \
            'tract_to_msa_md_income_pct', \
            'number_of_owner_occupied_units', \
            'number_of_1_to_4_family_units']
cat_cols = ['msa_md', 'state_code', 'loan_type', 'property_type', 'loan_purpose', 'occupancy', 'preapproval', \
            'applicant_ethnicity', 'applicant_race', 'applicant_sex', 'co_applicant'] 


## Prepare the model matrix
## Categorical features

def encode_string(cat_feature):
    ## First encode the strings to numeric categories
    enc = preprocessing.LabelEncoder()
    enc.fit(cat_feature)
    enc_cat_features = enc.transform(cat_feature)
    ## Now, apply one hot encodeing
    ohe = preprocessing.OneHotEncoder(categories='auto')
    encoded = ohe.fit(enc_cat_features.reshape(-1,1))
    return encoded.transform(enc_cat_features.reshape(-1,1)).toarray()

def trans_features(hmda_train):
    for col in cat_cols:
        temp = encode_string(hmda_train[col])
        if (col == cat_cols[0]):
            Features = temp
        else:
            Features = np.concatenate([Features, temp], axis = 1)

    print('Transformed categorical features')
    ncal=Features.shape[1]
    print('ncal', ncal)
    print(Features.shape)
    print(Features[:2,:])

    ## Numeric features
    Features = np.concatenate([Features, np.array(hmda_train[num_cols])], axis = 1)
    print('Transformed features')
    print(Features.shape)
    print(Features[:2,:])
    return Features, ncal

Features, ncal = trans_features(hmda_train)

## Split the dataset
nr.seed(1234)
labels = np.array(hmda_train['rate_spread'])
indx = range(Features.shape[0])
indx = ms.train_test_split(indx, test_size = 20000)
X_train = Features[indx[0],:]
y_train = np.ravel(labels[indx[0]])
X_test = Features[indx[1],:]
y_test = np.ravel(labels[indx[1]])

## Rescale numeric features
scaler = preprocessing.StandardScaler().fit(X_train[:,ncal:])
X_train[:,ncal:] = scaler.transform(X_train[:,ncal:])
X_test[:,ncal:] = scaler.transform(X_test[:,ncal:])
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
dump(lin_mod, 'SimplestLinearModel_l2.joblib')


######################################################################
## Apply l2 regularization
def plot_regularization(l, train_RMSE, test_RMSE, coefs, min_idx, title):
    plt.plot(l, test_RMSE, color = 'red', label = 'Test RMSE')
    plt.plot(l, train_RMSE, label = 'Train RMSE')
    plt.axvline(min_idx, color = 'black', linestyle = '--')
    plt.legend()
    plt.xlabel('Regularization parameter')
    plt.ylabel('Root Mean Square Error')
    plt.title(title)
    plt.show()
    
    plt.plot(l, coefs)
    plt.axvline(min_idx, color = 'black', linestyle = '--')
    plt.title('Model coefficient values \n vs. regularization parameter')
    plt.xlabel('Regularization parameter')
    plt.ylabel('Model coefficient value')
    plt.show()

def test_regularization_l2(X_train, y_train, X_test, y_test, l2):
    train_RMSE = []
    test_RMSE = []
    coefs = []
    for reg in l2:
        lin_mod = linear_model.Ridge(alpha = reg)
        lin_mod.fit(X_train, y_train)
        coefs.append(lin_mod.coef_)
        y_score_train = lin_mod.predict(X_train)
        train_RMSE.append(sklm.mean_squared_error(y_train, y_score_train))
        y_score = lin_mod.predict(X_test)
        test_RMSE.append(sklm.mean_squared_error(y_test, y_score))
    min_idx = np.argmin(test_RMSE)
    min_l2 = l2[min_idx]
    min_RMSE = test_RMSE[min_idx] 

    title = 'Train and test root mean square error \n vs. regularization parameter'
    plot_regularization(l2, train_RMSE, test_RMSE, coefs, min_l2, title)
    return min_l2, min_RMSE

l2 = [x for x in range(1,101)]
out_l2 = test_regularization_l2(X_train, y_train, X_test, y_test, l2)
print(out_l2)

#####################################################################
lin_mod_l2 = linear_model.Ridge(alpha = out_l2[0])
lin_mod_l2.fit(X_train, y_train)
y_score_l2 = lin_mod_l2.predict(X_test)

print_metrics(y_test, y_score_l2, 501)
hist_resids(y_test, y_score_l2)
resid_qq(y_test, y_score_l2)
resid_plot(y_test, y_score_l2)
