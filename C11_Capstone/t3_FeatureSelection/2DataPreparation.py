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
cat_cols = ['msa_md', 'state_code', \
            'county_code', 'lender', \
            'loan_type', 'property_type', 'loan_purpose', 'occupancy', 'preapproval', \
            'applicant_ethnicity', 'applicant_race', 'applicant_sex', 'co_applicant'] 
num_cols = ['loan_amount', 'applicant_income', \
            'population', 'minority_population_pct', 'ffiecmedian_family_income', \
            'tract_to_msa_md_income_pct', \
            'number_of_owner_occupied_units', \
            'number_of_1_to_4_family_units']

##  Identity row_id
##  Label    rate_spread

print(hmda_train.head())
print(hmda_train.shape)
print(hmda_train.columns)
print(hmda_train.dtypes)


## Prepare the model matrix
## Categorical features

def encode_string(cat_feature, col_name):
    ## First encode the strings to numeric categories
    enc = preprocessing.LabelEncoder()
    enc.fit(cat_feature)
    enc.classes_ = np.append(enc.classes_, -2)
    enc_cat_features = enc.transform(cat_feature)
    ## Now, apply one hot encodeing
    ohe = preprocessing.OneHotEncoder(categories='auto')
    encoded = ohe.fit(enc_cat_features.reshape(-1,1))
    col_names = ohe.get_feature_names(input_features = [col_name])
    print('feature names')
    print(ohe.get_feature_names(input_features = [col_name]))
    return encoded.transform(enc_cat_features.reshape(-1,1)).toarray(), col_names

def trans_features(hmda_train):
    for col in cat_cols:
        temp, temp_col_names = encode_string(hmda_train[col], col)
        print(col, temp.shape[1])
        if (col == cat_cols[0]):
            Features = temp
            Columns = temp_col_names
        else:
            Features = np.concatenate([Features, temp], axis = 1)
            Columns = np.concatenate([Columns, temp_col_names])

    print('Transformed categorical features')
    ncal=Features.shape[1]
    print('ncal', ncal)
    print(Features.shape)
    print(Features[:2,:])

    ## Numeric features
    Features = np.concatenate([Features, np.array(hmda_train[num_cols])], axis = 1)
    Columns = np.concatenate([Columns, num_cols])
    print('Transformed features')
    print(Features.shape)
    print(Features[:2,:])
    return Features, ncal, Columns

Features, ncal, Columns = trans_features(hmda_train)

## Rescale numeric features
scaler = preprocessing.StandardScaler().fit(Features[:,ncal:])
Features[:,ncal:] = scaler.transform(Features[:,ncal:])
print('Features for ML')
print(Features.shape)
print(Features[:5,:])

hmda_features = pd.DataFrame(Features, columns = Columns)
#hmda_features.head(5).to_csv('hmda_features_w_col_names.csv', index=False)
hmda_features.to_csv('hmda_features.csv', index=False)
hmda_train[['rate_spread']].to_csv('hmda_labels.csv', index=False)

#####################################################################
## Prepare the model matrix for test dataset

hmda_test = pd.read_csv('test_values.csv')
print('Prepare the model matrix for test dataset')
print(hmda_test.columns)
print(hmda_test.shape)
def encode_string_t(cat_feature, train_feature):
    ## First encode the strings to numeric categories
    enc = preprocessing.LabelEncoder()
    enc.fit(train_feature)
    cat_feature = cat_feature.map(lambda s: -2 if s not in enc.classes_ else s)
    enc.classes_ = np.append(enc.classes_, -2)
    enc_train_features = enc.transform(train_feature)
    enc_cat_features = enc.transform(cat_feature)
    ## Now, apply one hot encodeing
    ohe = preprocessing.OneHotEncoder(categories='auto')
    encoded_train = ohe.fit(enc_train_features.reshape(-1,1))
    ohe_t = preprocessing.OneHotEncoder(categories=ohe.categories_)
    encoded = ohe_t.fit(enc_cat_features.reshape(-1,1))
    return encoded.transform(enc_cat_features.reshape(-1,1)).toarray()

def impute_hmda_data(hmda_test, hmda_train):
    from sklearn.impute import SimpleImputer
    ## Recode names
    cols = hmda_test.columns
    hmda_test.columns = [str.replace('-','_') for str in cols]
    ## 'property_type = 3' does not appear in training set
    hmda_test.loc[hmda_test['property_type']==3.0, 'property_type'] = -1
    ### 'county_code' [63.0, 79.0, 85.0, 129.0, 132.0, 197.0, 203.0, 212.0, 289.0, 296.0, 298.0, 316.0]
    #for id_county_code in  [63.0, 79.0, 85.0, 129.0, 132.0, 197.0, 203.0, 212.0, 289.0, 296.0, 298.0, 316.0]:
    #    hmda_test.loc[hmda_test['county_code'] == id_county_code, 'county_code'] = 8
    ## Count missing values
    for column in cat_cols:
        print(column, hmda_test.loc[hmda_test[column] == -1, column].shape[0])
    for column in num_cols:
        print(column, hmda_test.loc[hmda_test[column].isna() , column].shape[0])
    ## Imputation of missing values using scikit-learn
    hmda_test['rate_spread'] = 1.0
    imp_n = SimpleImputer(missing_values = np.nan, strategy = 'median')
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
        print(col, temp.shape[1])
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

hmda_features_new = pd.DataFrame(Features_new, columns = Columns)
hmda_features_new.to_csv('hmda_features_new.csv', index=False)
