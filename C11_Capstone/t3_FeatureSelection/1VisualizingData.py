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

#####################################################################

## Explore the data
#print(hmda_train.head())
#print(hmda_train.dtypes)
#for column in num_cols:
#    print(hmda_train[column].describe())
#print(hmda_train['rate_spread'].describe())

## Compute and display a frequency table
def count_unique(hmda_train, cols):
    for col in cols:
        print('\n' + 'For column ' + col)
        print(len(np.array(hmda_train[col].unique())))
        print(hmda_train[col].value_counts())
        hmda_vc=hmda_train[col].value_counts()#.sort_values(by=col, ascending=True)
        print(hmda_vc.sort_index())
  

#count_unique(hmda_train, cat_cols + ['rate_spread'])
## Category columns with some categories having few distributions
##     'county_code', 'lender'
##     'msa_md', 'state_code'  ???


## Treat outliers
##   'rate_spread' = 99 (3) or <= 32
#hmda_outliers = hmda_train.loc[hmda_train['rate_spread'] == 99. ]
#hmda_outliers = hmda_train.loc[hmda_train['rate_spread'] >= 9. ]
#print(hmda_outliers.shape)
#for col in hmda_outliers.columns:
#    print(hmda_outliers[col])
#hmda_train.loc[hmda_train['rate_spread'] == 99, 'rate_spread'] = np.nan
#hmda_train.loc[hmda_train['rate_spread'] >= 9, 'rate_spread'] = np.nan
#hmda_train.dropna(axis = 0, inplace = True)
#print(hmda_train.shape)


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

## * Bar charts dropping low frequency entries
def plot_bars_m(hmda_train, cols):
    for col in cols:
        fig = plt.figure(figsize=(6,6)) # define plot area
        ax = fig.gca() # define axis
        counts = hmda_train[col].value_counts() # find the counts for each unique category
        counts_red = counts.loc[counts > 0.01 * hmda_train[col].shape[0]].sort_index()
        counts_low = counts.loc[counts <= 0.01 * hmda_train[col].shape[0]].sort_index()
        #print(col)
        #print(type(counts), counts.shape)
        #print(type(counts_red), counts_red.shape)
        #print(hmda_train[col].shape[0])
        #print(counts_red)
        counts_red.plot.bar(ax = ax, color = 'blue') # Use the plot.bar method on the counts data frame
        ax.set_title('Counts by ' + col) # Give the plot a main title
        ax.set_xlabel(col) # Set text for the x axis
        ax.set_ylabel('Counts')# Set text for y axis
        ax.ticklabel_format(axis='y', style='sci', scilimits = (0,0))
        fig.tight_layout()
        fig.savefig(col + '_counts.png',dpi=600)
        plt.show()


#plot_bars_m(hmda_train, cat_cols)
#plot_bars_m(hmda_train, ['rate_spread'])

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
        fig = plt.figure(figsize=(6,6)) # define plot area
        ax = fig.gca() # define axis    
        sns.set_style("whitegrid")
        sns.distplot(hmda_train[col], bins = bins, rug=True, hist = hist)
        plt.title('Histogram of ' + col)
        plt.xlabel(col)
        plt.ylabel('Counts')
        fig.tight_layout()
        fig.savefig(col + '_KDE.png',dpi=600)
        plt.show()

## * Histograms and KDE
#plot_density_hist(hmda_train, num_cols, bins = 100, hist = True)
#plot_density_hist(hmda_train, ['rate_spread'], bins = 100, hist = True)

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

#plot_density_2d(hmda_train, num_cols)

## Relation between categorical and numeric variables
## * Box plots
def plot_box(hmda_train, cols, col_y = 'rate_spread'):
    for col in cols:
        sns.set_style("whitegrid")
        counts = hmda_train[col].value_counts() # find the counts for each unique category
        counts_red = counts.loc[counts > 0.01 * hmda_train[col].shape[0]].sort_index().index
        index_red = np.ravel(counts_red)
        print(col,index_red.shape, index_red)
        hmda_train_red = hmda_train.loc[hmda_train[col].isin(index_red),[col,col_y]]
        sns.boxplot(col, col_y, data=hmda_train_red)
        plt.xlabel(col)
        plt.ylabel(col_y)
        plt.show()

#plot_box(hmda_train, cat_cols)

## * Violine plots
def plot_violin(hmda_train, cols, col_y = 'rate_spread'):
    for col in cols:
        counts = hmda_train[col].value_counts() # find the counts for each unique category
        index_red = counts_red = counts.loc[counts > 0.01 * hmda_train[col].shape[0]].sort_index().index
        print(index_red)
        sns.set_style("whitegrid")
        hmda_train_red = hmda_train.loc[hmda_train[col] in index_red ,[col, col_y]]
        sns.violinplot(col, col_y, data=hmda_train_red)
        plt.xlabel(col)
        plt.ylabel(col_y)
        plt.show()

#plot_violin(hmda_train, cat_cols)

## Additional dimensions

## Pair-wise scatter plot
def plot_correlation(hmda_train, cols ):
    snspg = sns.pairplot(hmda_train, vars = cols, diag_kind="auto") #, height=2) #, aspect=1)
    #snspg.map_upper(sns.kdeplot, cmap="Blues_d")
    for ax in snspg.axes.flat:
        ax.ticklabel_format(axis='both', style='sci', scilimits = (0,0))
    plt.tight_layout()
    plt.savefig('hmda_pairwise.png',dpi=600)
    plt.show()

num_cols_sub1 = ['loan_amount', 'applicant_income']
num_cols_sub2 = ['population', 'minority_population_pct', 'ffiecmedian_family_income', \
            'tract_to_msa_md_income_pct', 'number_of_owner_occupied_units', \
            'number_of_1_to_4_family_units']
fm = open('tab_corr.txt', "w")
corr = hmda_train[num_cols].corr()
print(corr)
for i in range(corr.shape[0]):
    fm.write(" %s "%(corr.index[i]))
    for j in range(corr.shape[1]):
        fm.write(" %f "%(np.array(corr)[i][j]))
fm.close()
plot_correlation(hmda_train, num_cols_sub1)
