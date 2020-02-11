## Data preparation using scikit-learn
## Labels
## Category: BikeBuyer
## Numeric:  AveMonthSpend --> log_AveMonthSpend
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import numpy.random as nr
import math

# Import data for customer information
aw = pd.read_csv('AW_join.csv')
print( 'Load AW_join.csv')

## Features for ML training
cat_cols = ['Education', 'Occupation', 'Gender', 'MaritalStatus', 'HomeOwnerFlag', 'NCarsGroup', 'NChildrenAtHomeGroup', 'TotalChildren']
num_cols = ['YearlyIncome', 'Age']

## Transform categorical features
##    Aggregating categorical variables
aw['TotalChildrenGroup'] = pd.cut(aw['TotalChildren'], \
          bins = [-1,0,1,2,3,4,100], labels = ['No','1','2','3','4','>=5'], right=True)

## Transform numerical features and the label
def hist_plot(vals, lab):
    ## Distribution plot of values
    sns.distplot(vals)
    plt.title('Histogram of ' + lab)
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.show()

## Labels AveMonthSpend
hist_plot(aw['AveMonthSpend'], 'AveMonthSpend')

aw['log_AveMonthSpend'] = np.log(aw['AveMonthSpend'])
hist_plot(aw['log_AveMonthSpend'], 'log AveMonthSpend')
num_cols = ['YearlyIncome', 'Age']
for col in num_cols:
    hist_plot(aw[col],col)
    aw['log_'+col] = np.log(aw[col])
    hist_plot(aw['log_'+col],'log_'+col)

## Countour plots / 2d density plots
def plot_density_2d(aw, cols, col_y = 'AveMonthSpend', kind = 'kde'):
    for col in cols:
        sns.set_style("whitegrid")
        sns.jointplot(col, col_y, data=aw, kind=kind)
        plt.xlabel(col)
        plt.ylabel(col_y)
        plt.show()

plot_density_2d(aw, num_cols)
plot_density_2d(aw, num_cols, col_y = 'log_AveMonthSpend')


cat_cols = ['Education', 'Occupation', 'Gender', 'MaritalStatus', 'HomeOwnerFlag', 'NCarsGroup', 'NChildrenAtHomeGroup', 'TotalChildrenGroup']
num_cols = ['YearlyIncome', 'Age']
aw_prep = aw[cat_cols + num_cols + ['AveMonthSpend', 'log_AveMonthSpend', 'BikeBuyer']]

aw_prep.to_csv('AW_prep.csv', index=False)
print( aw_prep.columns )
print( aw_prep.dtypes)
print( aw_prep.shape )
