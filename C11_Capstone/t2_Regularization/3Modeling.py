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
Features = np.array(pd.read_csv('hmda_features_prep.csv'))
Labels = np.array(pd.read_csv('hmda_labels_prep.csv'))
print('Load Features and Labels')
print(Features.shape)
print(Labels.shape)
"""
hmda_train = pd.read_csv('hmda_train.csv')
print('Load hmda_train.csv')
print(hmda_train.shape)

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
"""

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

"""
## Rescale numeric features
scaler = preprocessing.StandardScaler().fit(X_train[:,ncal:])
X_train[:,ncal:] = scaler.transform(X_train[:,ncal:])
X_test[:,ncal:] = scaler.transform(X_test[:,ncal:])
print(X_train.shape)
print(X_train[:5,:])
"""

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
#dump(scaler, 'SimplestLinearModel_scaler.joblib')


"""
######################################################################
## Predict new input features
hmda_test = pd.read_csv('test_values.csv')
print(hmda_test.columns)
print(hmda_test.shape)
def encode_string_t(cat_feature, train_feature):
    ## First encode the strings to numeric categories
    enc = preprocessing.LabelEncoder()
    enc.fit(train_feature)
    enc_cat_features = enc.transform(cat_feature)
    ## Now, apply one hot encodeing
    ohe = preprocessing.OneHotEncoder(categories='auto')
    encoded = ohe.fit(enc_cat_features.reshape(-1,1))
    return encoded.transform(enc_cat_features.reshape(-1,1)).toarray()

def impute_hmda_data(hmda_test, hmda_train):
    from sklearn.impute import SimpleImputer
    ## Recode names
    cols = hmda_test.columns
    hmda_test.columns = [str.replace('-','_') for str in cols]
    hmda_test.loc[hmda_test['property_type']==3.0, 'property_type'] = -1
    ## Count missing values
    for column in cat_cols:
        print(column, hmda_test.loc[hmda_test[column] == -1, column].shape[0])
    for column in num_cols:
        print(column, hmda_test.loc[hmda_test[column].isna() , column].shape[0])
    ## Imputation of missing values using scikit-learn
    hmda_test['rate_spread'] = 1.0
    imp_n = SimpleImputer(missing_values = np.nan, strategy = 'mean')
    imp_n.fit(hmda_train)
    hmda_test = imp_n.transform(hmda_test)
    imp_c = SimpleImputer(missing_values = -1, strategy = 'most_frequent')
    imp_c.fit(hmda_train)
    hmda_test = imp_c.transform(hmda_test)
    print(type(hmda_test))
    hmda_test = pd.DataFrame(hmda_test, columns=hmda_train.columns)
    print(type(hmda_test))
    hmda_test.to_csv('hmda_test_imputed.csv', index = False)
    for column in cat_cols:
        print(column, hmda_test.loc[hmda_test[column] == -1, column].shape[0])
    for column in num_cols:
        print(column, hmda_test.loc[hmda_test[column].isna() , column].shape[0])

    ## Encode categorical features
    for col in cat_cols:
        print(col)
        temp = encode_string_t(hmda_test[col], train_feature = hmda_train[col])
        if(col == cat_cols[0]):
            Features = temp
        else:
            Features = np.concatenate([Features, temp], axis = 1)
    ncal=Features.shape[1]
    ## Add numeric features
    Features = np.concatenate([Features, np.array(hmda_test[num_cols])], axis = 1)
    print('Transformed features for new dataset')
    print(Features.shape)
    print(Features[:20,:])
    return Features

X_new = impute_hmda_data(hmda_test, hmda_train)
X_new[:,ncal:] = scaler.transform(X_new[:,ncal:])
print(X_new.shape)
print(X_new[:5,:])
y_predict = lin_mod.predict(X_new).round(1)
print(type(y_predict))
print(len(y_predict))

hmda_pred = pd.DataFrame({'row_id': np.array(hmda_test['row_id']), \
                          'rate_spread': y_predict[:]})
hmda_pred.to_csv('hmda_pred.csv', index=False)

"""
