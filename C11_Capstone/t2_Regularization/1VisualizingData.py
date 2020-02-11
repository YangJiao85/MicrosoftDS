# Visualize and explore data / exploratory data analysis
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

## Import data
hmda_train=pd.read_csv('hmda_train.csv')
cat_cols = ['msa_md', 'state_code', 'county_code', \
            'lender', 'loan_type', 'property_type', 'loan_purpose', 'occupancy', 'preapproval', \
            'applicant_ethnicity', 'applicant_race', 'applicant_sex', 'co_applicant'] 
num_cols = ['loan_amount', 'applicant_income', \
            'population', 'minority_population_pct', 'ffiecmedian_family_income', \
            'tract_to_msa_md_income_pct', 'number_of_owner_occupied_units', \
            'number_of_1_to_4_family_units']
##  Identity row_id
##  Label    rate_spread

## Accumulate categories
## Recode the categorical features

code_list = [
    ['loan_type',
      { 1 : 'Conventional',
        2 : 'FHA-insured',
        3 : 'VA-guaranteed',
        4 : 'FSA/RHS' } ], 
    ['property_type',
      { 1 : '1-4-family',
        2 : 'Manufactured housing',
        3 : 'Multifamily' } ],
    ['loan_purpose',
      { 1 : 'Home purchase',
        2 : 'Home improvement',
        3 : 'Refinancing' } ],
    ['occupancy' ,
      { 1 : 'Owner-occupied',
        2 : 'Not owner-occupied',
        3 : 'Not applicable' } ],
    ['preapproval',
      { 1 : 'Requested',
        2 : 'Not requested',
        3 : 'Not applicable' } ],
    ['applicant_ethnicity',
      { 1 : 'Hispanic or Latino',
        2 : 'Not Hispanic or Latino',
        3 : 'Not provided',
        4 : 'Not applicable' } ],
    ['applicant_race',
      { 1 : 'American Indian or Alaska Native',
        2 : 'Asian',
        3 : 'Black or African American',
        4 : 'Native Hawaiian or Other Pacific Islander',
        5 : 'White',
        6 : 'Not provided',
        7 : 'Not applicable' } ],
    ['applicant_sex',
      { 1 : 'Male',
        2 : 'Female',
        3 : 'NA',
        4 : 'Not applicable' } ]
]

for col_dic in code_list:
    col = col_dic[0]
    dic = col_dic[1]
    hmda_train[col] = [dic[x] for x in hmda_train[col]]

print(hmda_train.head())

#####################################################################

## Explore the data
print(hmda_train.head())
print(hmda_train.dtypes)
for column in num_cols:
    print(hmda_train[column].describe())
print(hmda_train['rate_spread'].describe())

## Compute and display a frequency table
def count_unique(hmda_train, cols):
    for col in cols:
        print('\n' + 'For column ' + col)
        print(hmda_train[col].value_counts())
        hmda_vc=hmda_train[col].value_counts()#.sort_values(by=col, ascending=True)
        print(hmda_vc.sort_index())
  

count_unique(hmda_train, cat_cols + ['rate_spread'])
## Category columns with some categories having few distributions
##     'county_code', 'lender'
##     'msa_md', 'state_code'  ???

cat_cols = ['msa_md', 'state_code', 'loan_type', 'property_type', 'loan_purpose', 'occupancy', 'preapproval', \
            'applicant_ethnicity', 'applicant_race', 'applicant_sex', 'co_applicant'] 

## Treat outliers
##   'rate_spread' = 99 (3) or <= 32

#hmda_outliers = hmda_train.loc[hmda_train['rate_spread'] == 99. ]
hmda_outliers = hmda_train.loc[hmda_train['rate_spread'] >= 9. ]
print(hmda_outliers.shape)
#for col in hmda_outliers.columns:
#    print(hmda_outliers[col])
#hmda_train.loc[hmda_train['rate_spread'] == 99, 'rate_spread'] = np.nan
hmda_train.loc[hmda_train['rate_spread'] >= 9, 'rate_spread'] = np.nan
hmda_train.dropna(axis = 0, inplace = True)
print(hmda_train.shape)


## Visualizing data
## Visualizing distributions (1D)
## * Bar charts
def plot_bars(hmda_train, cols):
    for col in cols:
        fig = plt.figure(figsize=(6,6)) # define plot area
        ax = fig.gca() # define axis    
        counts = hmda_train[col].value_counts() # find the counts for each unique category
        counts.plot.bar(ax = ax, color = 'blue') # Use the plot.bar method on the counts data frame
        ax.set_title('Counts by ' + col) # Give the plot a main title
        ax.set_xlabel(col) # Set text for the x axis
        ax.set_ylabel('Counts')# Set text for y axis
        plt.show()

#plot_bars(hmda_train, cat_cols)

## * Histograms
def plot_histogram(hmda_train, cols, bins = 10):
    for col in cols:
        fig = plt.figure(figsize=(6,6))
        ax = fig.gca()
        hmda_train[col].plot.hist(ax = ax, bins = bins)
        ax.set_title('Histogram of ' + col) 
        ax.set_xlabel(col)
        ax.set_ylabel('Counts')
        plt.show()

#plot_histogram(hmda_train, num_cols)

## * KDE (kernel density estimation) using Seaborn
def plot_density_hist(hmda_train, cols, bins = 10, hist = False):
    for col in cols:
        sns.set_style("whitegrid")
        sns.distplot(hmda_train[col], bins = bins, rug=True, hist = hist)
        plt.title('Histogram of ' + col)
        plt.xlabel(col)
        plt.ylabel('Counts')
        plt.show()

## * Histograms and KDE
#plot_density_hist(hmda_train, num_cols, bins = 20, hist = True)

## Two dimensional plots
## Scatter
def plot_scatter(hmda_train, cols, col_y = 'rate_spread'):
    for col in cols:
        fig = plt.figure(figsize=(7,6))
        ax = fig.gca()
        hmda_train.plot.scatter(x = col, y = col_y, ax = ax)
        ax.set_title('Scatter plot of ' + col_y + ' vs. ' + col)
        ax.set_xlabel(col)
        ax.set_ylabel(col_y)
        plt.show()

#plot_scatter(hmda_train, num_cols)

## Check colinear relation

## Deal with overplotting
## * Transparency
def plot_scatter_t(hmda_train, cols, col_y = 'rate_spread', alpha=1.0):
    for col in cols:
        fig = plt.figure(figsize=(7,6))
        ax = fig.gca()
        hmda_train.plot.scatter(x = col, y = col_y, ax = ax, alpha = alpha)
        ax.set_title('Scatter plot of ' + col_y + ' vs. ' + col)
        ax.set_xlabel(col)
        ax.set_ylabel(col_y)
        plt.show()

#plot_scatter_t(hmda_train, num_cols, alpha = 0.2)

## * Countour plots / 2d density plots
def plot_density_2d(hmda_train, cols, col_y = 'rate_spread', kind = 'kde'):
    for col in cols:
        sns.set_style("whitegrid")
        sns.jointplot(col, col_y, data=hmda_train, kind = kind)
        plt.xlabel(col)
        plt.ylabel(col_y)
        plt.show()

plot_density_2d(hmda_train, num_cols)

## Relation between categorical and numeric variables
## * Box plots
def plot_box(hmda_train, cols, col_y = 'rate_spread'):
    for col in cols:
        sns.set_style("whitegrid")
        sns.boxplot(col, col_y, data=hmda_train)
        plt.xlabel(col)
        plt.ylabel(col_y)
        plt.show()

#plot_box(hmda_train, cat_cols)

## * Violine plots
def plot_violin(hmda_train, cols, col_y = 'rate_spread'):
    for col in cols:
        sns.set_style("whitegrid")
        sns.violinplot(col, col_y, data=hmda_train)
        plt.xlabel(col)
        plt.ylabel(col_y)
        plt.show()

plot_violin(hmda_train, cat_cols)

## Additional dimensions

hmda_train.to_csv('hmda_train_pre1.csv', index=False)
