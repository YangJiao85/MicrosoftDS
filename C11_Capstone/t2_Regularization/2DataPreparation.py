##
## Labels
## Numeric:   rate_spread
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
import numpy as np
import numpy.random as nr
import math

## Import data
hmda_train=pd.read_csv('hmda_train.csv')
print('Load hmda_train.csv')
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

##  Identity row_id
##  Label    rate_spread

print(hmda_train.head())
print(hmda_train.shape)
print(hmda_train.columns)
print(hmda_train.dtypes)


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

## Rescale numeric features
scaler = preprocessing.StandardScaler().fit(Features[:,ncal:])
Features[:,ncal:] = scaler.transform(Features[:,ncal:])
print('Features for ML')
print(Features.shape)
print(Features[:5,:])

hmda_features = pd.DataFrame(Features)
#hmda_features.to_csv('hmda_features_prep.csv', index=False)
#hmda_train[['rate_spread']].to_csv('hmda_labels_prep.csv', index=False)

#####################################################################
## Prepare the model matrix for test dataset

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
    #hmda_test.to_csv('hmda_test_imputed.csv', index = False)
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
    return Features, ncal

Features_new, ncal = impute_hmda_data(hmda_test, hmda_train)
## Rescale numeric features
Features_new[:,ncal:] = scaler.transform(Features_new[:,ncal:])
print(Features_new.shape)
print(Features_new[:5,:])

hmda_features_new = pd.DataFrame(Features_new)
hmda_features_new.to_csv('hmda_features_new_prep.csv', index=False)
