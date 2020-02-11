# Visualize and explore data for classification
# Label: Bike buyer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import numpy.random as nr
import math
import seaborn as sns

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
#cat_cols = ['Education', 'Occupation', 'Gender', 'MaritalStatus', 'HomeOwnerFlag', 'NCarsGroup', 'NChildrenAtHomeGroup']
cat_cols = ['Occupation', 'Gender', 'MaritalStatus', 'NCarsGroup', 'NChildrenAtHomeGroup']
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
NChildrenAtHomeGroup     object
BikeBuyer                 int64
dtype: object
(16404, 25)
(16404,)
"""

## Examine classes and class imbalance
BikeBuyer_counts = aw['BikeBuyer'].value_counts()
print(BikeBuyer_counts)
##
##  0    10949
##  1     5455
##

## Class separation quality of numeric features
def plot_violin(aw, cols, col_x = 'BikeBuyer'):
    for col in cols:
        sns.set_style("whitegrid")
        sns.violinplot(col_x, col, data=aw)
        plt.xlabel(col_x)
        plt.ylabel(col)
        plt.show()

plot_violin(aw, num_cols)


## Class separation quality of categorical features
aw['dummy'] = np.ones(shape = aw.shape[0])
for col in cat_cols:
    print(col)
    counts = aw[['dummy', 'BikeBuyer', col]].groupby(['BikeBuyer', col], as_index = False).count()
    temp = counts[counts['BikeBuyer'] == 0][[col, 'dummy']]
    _ = plt.figure(figsize = (10,4))
    plt.subplot(1, 2, 1)
    temp = counts[counts['BikeBuyer'] == 0][[col, 'dummy']]
    plt.bar(temp[col], temp.dummy)
    plt.xticks(rotation=90)
    plt.title('Counts for ' + col + '\n NO Bike Buyer')
    plt.ylabel('count')
    plt.subplot(1, 2, 2)
    temp = counts[counts['BikeBuyer'] == 1][[col, 'dummy']]
    plt.bar(temp[col], temp.dummy)
    plt.xticks(rotation=90)
    plt.title('Counts for ' + col + '\n Bike Buyer')
    plt.ylabel('count')
    plt.show()

