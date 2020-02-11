# Visualize and explore data for regression
# Label: AveMonthSpend
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import numpy.random as nr
import math

# Import data for customer information
aw = pd.read_csv('AW_join.csv')
print( 'Load AW_join.csv')
print( aw.columns )
print( aw.dtypes )
print( aw.shape )
print( aw['CustomerID'].unique().shape )
print( aw.head() )

## Features
##   CustomerID FirstName LastName --> identity drop
##   AddressLine1                               drop
##   City StateProvinceName CountryRegionName PostalCode --> Location  ?
##   CountryRegionName  -->   weak feature
##   PhoneNumber                                drop
##   BirthDate Age AgeGroup        -->  keep AgeGroup
##   Education Occupation Gender MaritalStatus 
##   HomeOwnerFlag  -->       no feature
##
##   Numerical features
##     YearlyIncome  
##     NumberCarsOwned NumberChildrenAtHome TotalChildren 
##     Age
##   Categorical features
##     HomeOwnerFlag
##     Occupation  Gender  MaritalStatus 
##     Education                           --> weak feature
##     AgeGroup NCarsGroup NChildrenAtHomeGroup
## Labels
## Category: BikeBuyer
## Numeric:  AveMonthSpend

## Columns for plot
cat_cols = ['CountryRegionName', 'Education', 'Occupation', 'Gender', 'MaritalStatus', 'HomeOwnerFlag', 'AgeGroup', 'NCarsGroup', 'NChildrenAtHomeGroup']
num_cols = ['NumberCarsOwned', 'NumberChildrenAtHome', 'TotalChildren', 'YearlyIncome', 'Age' ]
## Features for ML training
cat_cols = ['Education', 'Occupation', 'Gender', 'MaritalStatus', 'HomeOwnerFlag', 'NCarsGroup', 'NChildrenAtHomeGroup']
num_cols = ['TotalChildren', 'YearlyIncome', 'Age']

"""
Index(['CustomerID', 'FirstName', 'LastName', 'AddressLine1', 'City',
       'StateProvinceName', 'CountryRegionName', 'PostalCode', 'PhoneNumber',
       'BirthDate', 'Education', 'Occupation', 'Gender', 'MaritalStatus',
       'HomeOwnerFlag', 'NumberCarsOwned', 'NumberChildrenAtHome',
       'TotalChildren', 'YearlyIncome', 'Age', 'AgeGroup', 'AveMonthSpend',
       'NCarsGroup', 'NChildrenAtHomeGroup', 'BikeBuyer'],
      dtype='object')
CustomerID                int64
FirstName                object
LastName                 object
AddressLine1             object
City                     object
StateProvinceName        object
CountryRegionName        object
PostalCode               object
PhoneNumber              object
BirthDate                object
Education                object
Occupation               object
Gender                   object
MaritalStatus            object
HomeOwnerFlag             int64
NumberCarsOwned           int64
NumberChildrenAtHome      int64
TotalChildren             int64
YearlyIncome              int64
Age                     float64
AgeGroup                 object
AveMonthSpend             int64
NCarsGroup               object
NChildrenAtHomeGroup           object
BikeBuyer                 int64
dtype: object
(16404, 25)
(16404,)
"""

## Compute and display a frequency table
def count_unique(aw, cols):
    for col in cols:
        print('\n' + 'For column ' + col)
        print(aw[col].value_counts())

#print(count_unique(aw, cat_cols))

## Visualizing data
## Visualizing distribution
## * Bar charts
def plot_bars(aw, cols):
    for col in cols:
        fig = plt.figure(figsize=(6,6))
        ax = fig.gca()
        counts = aw[col].value_counts()   # find the counts for each unique category
        counts.plot.bar(ax = ax, color = 'blue') 
        ax.set_title('Number of customers by ' + col) 
        ax.set_xlabel(col) 
        ax.set_ylabel('Number fo customers')
        plt.show()

#plot_bars(aw, cat_cols)

def plot_histogram(aw, cols, bins=10):
    for col in cols:
        fig = plt.figure(figsize=(6,6))
        ax = fig.gca()
        nval = aw[col].unique().shape[0]
        aw[col].plot.hist(ax = ax, bins = min(nval,bins))
        ax.set_title('Histogram of ' + col)
        ax.set_xlabel(col)
        ax.set_ylabel('Number of customers')
        plt.show()

#plot_histogram(aw, num_cols)

## Visualize relation
def plot_scatter(aw, cols, col_y = 'AveMonthSpend'):
    for col in cols:
        fig = plt.figure(figsize=(8,6))
        ax = fig.gca()
        aw.plot.scatter(x = col, y = col_y, ax = ax)
        ax.set_title('Scatter plot of ' + col_y + ' vs. ' + col)
        ax.set_xlabel(col)
        ax.set_ylabel(col_y)
        plt.show()

#plot_scatter(aw, num_cols)

## Check colinear relation
#plot_scatter(aw, ['NumberChildrenAtHome'], 'TotalChildren')

## Countour plots / 2d density plots
def plot_density_2d(aw, cols, col_y = 'AveMonthSpend', kind = 'kde'):
    for col in cols:
        sns.set_style("whitegrid")
        sns.jointplot(col, col_y, data=aw, kind=kind)
        plt.xlabel(col)
        plt.ylabel(col_y)
        plt.show()

plot_density_2d(aw, num_cols)

## Hexbin plots
##plot_density_2d(aw, num_cols, kind = 'hex')

## Relation between categorical and numerical variables
## Violine plots
def plot_violin(aw, cols, col_y = 'AveMonthSpend'):
    for col in cols:
        sns.set_style("whitegrid")
        sns.violinplot(col, col_y, data=aw)
        plt.xlabel(col)
        plt.ylabel(col_y)
        plt.show()
        plt.savefig("violin_"+col_y+"_"+col+".png")

plot_violin(aw, cat_cols)

## Multi-axis views of data
