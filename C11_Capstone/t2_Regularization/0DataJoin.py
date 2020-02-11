##
#
# Analyse data from
#   train_values.csv
#   train_labels_abiUmgM.csv
#   test_values.csv               
#   submission_format_22TpNFD.csv 
# Target variable
#   rate_spread

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import numpy.random as nr

# Import data from train_values.csv
hmda_train_values = pd.read_csv('train_values.csv')
print('Load train_values.csv')
print(hmda_train_values.head(4))
print(hmda_train_values.columns)
print(hmda_train_values.dtypes)
print(hmda_train_values.shape)
print(hmda_train_values['row_id'].unique().shape[0])

# Import data from train_labels_abiUmgM.csv
hmda_train_labels = pd.read_csv('train_labels_abiUmgM.csv')
print('Load train_labels_abiUmgM.csv')
print(hmda_train_labels.head(10))
print(hmda_train_labels.columns)
print(hmda_train_labels.dtypes)
print(hmda_train_labels.shape)
print(hmda_train_labels['row_id'].unique().shape[0])

# Join hmda_train and hmda_labels on row_id
hmda_train = hmda_train_values.join(hmda_train_labels.set_index('row_id'), on='row_id', how='inner')
print(hmda_train.shape)
print(hmda_train.columns)
print(hmda_train.dtypes)

#(200000, 23)
#Index(['row_id', 'loan_type', 'property_type', 'loan_purpose', 'occupancy',
#       'loan_amount', 'preapproval', 'msa_md', 'state_code', 'county_code',
#       'applicant_ethnicity', 'applicant_race', 'applicant_sex',
#       'applicant_income', 'population', 'minority_population_pct',
#       'ffiecmedian_family_income', 'tract_to_msa_md_income_pct',
#       'number_of_owner-occupied_units', 'number_of_1_to_4_family_units',
#       'lender', 'co_applicant', 'rate_spread'],
#      dtype='object')

# Description
print(hmda_train[['rate_spread']].describe().round(1))

print(hmda_train[['applicant_ethnicity','rate_spread']].groupby('applicant_ethnicity').mean())
print(hmda_train[['applicant_sex','rate_spread']].groupby('applicant_sex').mean())

hmda_state43=hmda_train.loc[hmda_train['state_code'] == 43,['applicant_income','loan_amount']]
hmda_state48=hmda_train.loc[hmda_train['state_code'] == 48,['county_code','rate_spread']]

# Plot
def hist_plot(vals, lab, bins=100):
    ## Distribution plot
    sns.distplot(vals, bins=bins)
    plt.title('Histogram of ' + lab)
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.show()

#hist_plot(hmda_train['rate_spread'], 'rate_spread')

# Scatter
def plot_scatter(ds, cols, col_y = 'rate_spread'):
    for col in cols:
        fig = plt.figure(figsize=(8,6))
        ax = fig.gca()
        ds.plot.scatter(x = col, y = col_y, ax=ax)
        #ax.set_title('Scatter plot of ' + col_y + ' vs. ' + col)
        ax.set_xlabel(col)
        ax.set_ylabel(col_y)
        plt.show()

#plot_scatter(hmda_state43, ['applicant_income'], col_y = ['loan_amount'])
print(hmda_state43.shape)

hmda_state48_county=hmda_state48[['county_code','rate_spread']].groupby('county_code').mean()
print(hmda_state48[['county_code','rate_spread']].groupby('county_code').mean())
print(hmda_state48_county.describe())


hmda_st23_lt123=hmda_train.loc[hmda_train['loan_type'].isin([1,2,3]) \
                             & hmda_train['state_code'].isin([2,3])  ]

print(hmda_st23_lt123[['loan_type','state_code','rate_spread']].groupby('state_code').mean())
print(hmda_st23_lt123[['loan_type','state_code','rate_spread']].groupby(['loan_type','state_code']).mean())

cat_cols = ['msa_md', 'state_code', 'county_code', \
            'lender', 'loan_type', 'property_type', 'loan_purpose', 'occupancy', 'preapproval', \
            'applicant_ethnicity', 'applicant_race', 'applicant_sex', 'co_applicant'] 
num_cols = ['loan_amount', 'applicant_income', \
            'population', 'minority_population_pct', 'ffiecmedian_family_income', 'tract_to_msa_md_income_pct', 'number_of_owner_occupied_units', 'number_of_1_to_4_family_units']

#####################################################################

print('Number of missing msa_md')
print(hmda_train.loc[hmda_train['msa_md'] < 0, 'msa_md'].shape[0])

def clean_hmda_data(hmda_train, ifdrop = True):
    ## Recode names
    cols = hmda_train.columns
    hmda_train.columns = [str.replace('-','_') for str in cols]

    int_cols = ['row_id', 'loan_type', 'property_type', 'loan_purpose', 'occupancy',\
                'preapproval', 'msa_md', 'state_code', 'county_code', \
                'applicant_ethnicity', 'applicant_race', 'applicant_sex', 'lender']
    ## Treat missing values
    for column in cat_cols:
        print(column, hmda_train.loc[hmda_train[column] == -1, column].shape[0])
    for column in num_cols:
        print(column, hmda_train.loc[hmda_train[column].isna() , column].shape[0])
    if(ifdrop):
        for column in cat_cols:
            hmda_train.loc[hmda_train[column] == -1, column] = np.nan
        hmda_train.dropna(axis = 0, inplace = True)
    ## Transform column data type
    hmda_train['co_applicant'] = hmda_train['co_applicant'].astype('bool')
    for column in int_cols:
        hmda_train[column] = hmda_train[column].astype('int64')

print(hmda_train.shape)
print(hmda_train.dtypes)
clean_hmda_data(hmda_train)
print(hmda_train.shape)
print(hmda_train.dtypes)
# state_code 1338 missing values
#loan_amount 0
#applicant_income 10708
#population 1995
#minority_population_pct 1995
#ffiecmedian_family_income 1985
#tract_to_msa_md_income_pct 2023
#number_of_owner_occupied_units 2012
#number_of_1_to_4_family_units 2016

"""
print('test values')
hmda_test=pd.read_csv('test_values.csv')
print(hmda_test.shape)
clean_hmda_data(hmda_test)
print(hmda_test.shape)
"""


hmda_train.to_csv('hmda_train.csv', index = False)

