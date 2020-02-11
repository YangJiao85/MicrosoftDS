# Tasks
# Clean the data by replacing any missing values and removing duplicate rows.
#    In this dataset, each customer is identified by a unique customer ID.
#    The most recent version of a duplicated record should be retained.
# Explore the data by calculating summary and descriptive statistics for 
#    the features in the dataset, calculating correlations between features, 
#    and creating data visualizations to determine apparent relationships 
#    in the data.
# Based on your analysis of the customer data after removing all duplicate 
#    customer records, answer the questions below.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import numpy.random as nr
import math
import seaborn as sns

# Import data for customer information
aw_custs = pd.read_csv('AdvWorksCusts.csv')
print( 'Load AdvWorksCusts.csv')
print( aw_custs.head(20) )
print( aw_custs.columns )
print( aw_custs.dtypes)
#print( (aw_custs.astype(np.object) == np.nan).any() )
for col in aw_custs.columns:
   if aw_custs[col].dtype == object:
      count = 0
      count = [count + 1 for x in aw_custs[col] if type(x) == float]
      print(col + ' ' + str(sum(count)))
#for i in range(20):
#   print(aw_custs['MiddleName'][i], type(aw_custs['MiddleName'][i]))

# Drop column with too many missing values
aw_custs.drop(['Title', 'MiddleName', 'Suffix', 'AddressLine2'], axis = 1, inplace = True)
print(aw_custs.columns)
print(aw_custs.shape)
print(aw_custs['CustomerID'].unique().shape)
aw_custs.drop_duplicates(subset = 'CustomerID', keep = 'last', inplace = True)
print(aw_custs.shape)
aw_custs.to_csv('AdvWorksCusts_preped.csv', index=False)

aw_custs = pd.read_csv('AdvWorksCusts_preped.csv')
print( aw_custs.columns )
#print( aw_custs.groupby('Occupation').median() )
## new column 'age' by data collected date 1st January 1998 - 'BirthDate'
aw_custs['Age'] = (pd.to_datetime('1998-01-01') - pd.to_datetime(aw_custs['BirthDate'], errors='coerce')).astype('<m8[Y]')
## AgeGroup <25 , 25-45, 45-55, >55
aw_custs['AgeGroup'] = pd.cut(aw_custs['Age'], bins = [0,25,45,55,1000], \
          labels = ['<25','25-45','45-55','>55'], right=False)

# Import data from AveMonthSpend
aw_ams = pd.read_csv('AW_AveMonthSpend.csv')
print('Load AW_AveMonthSpend.csv')
print( aw_ams.columns )
print( aw_ams.dtypes )
print( aw_ams.shape )
print( aw_ams['CustomerID'].unique().shape )
aw_ams.drop_duplicates(subset = 'CustomerID', keep = 'last', inplace = True)
print( aw_ams.shape )
print( aw_ams['CustomerID'].unique().shape )
aw_ams.to_csv('AW_AveMonthSpend_Preped.csv', index=False)

# 
aw_ams = pd.read_csv('AW_AveMonthSpend_Preped.csv')
print(aw_ams.columns)

# Import data from AW_BikeBuyer.csv
aw_bb = pd.read_csv('AW_BikeBuyer.csv')
print('Load AW_BikeBuyer.csv')
print( aw_bb.columns )
print( aw_bb.dtypes )
print( aw_bb.shape )
print( aw_bb.CustomerID.unique().shape )
aw_bb.drop_duplicates(subset = 'CustomerID', keep = 'last', inplace = True)
print( aw_bb.shape )
print( aw_bb.CustomerID.unique().shape )
aw_bb.to_csv('AW_BikeBuyer_Preped.csv', index=False)

aw_bb = pd.read_csv('AW_BikeBuyer_Preped.csv')
print(aw_bb.describe())
aw_bb_counts = aw_bb['BikeBuyer'].value_counts()
print(aw_bb_counts)

# Join aw_custs aw_ams on CustomerID
aw_join=aw_custs.join(aw_ams.set_index('CustomerID'), on='CustomerID', how='inner')
print(aw_join.shape)
print(aw_join.columns)
print(aw_join.head(2))
print( aw_join.groupby(['Gender', 'AgeGroup']).mean() )
print( aw_join.groupby(['Gender', 'AgeGroup']).sum() )
print( aw_join.groupby('MaritalStatus').median() )
print( aw_join['NumberCarsOwned'].unique())
aw_join['NCarsGroup'] = pd.cut(aw_join['NumberCarsOwned'], \
          bins = [-1,0,2,10], labels = ['No','1-2','>=3'], right=True)
print( aw_join[['NumberCarsOwned','NCarsGroup']].head())
print( aw_join.groupby('NCarsGroup').median() )
print( aw_join[['Gender','AveMonthSpend']].groupby('Gender').describe() )
aw_join['NChildrenAtHomeGroup'] = pd.cut(aw_join['NumberChildrenAtHome'], \
          bins = [-1,0,100], labels = ['No','>=1'], right=True)
print( aw_join[['NChildrenAtHomeGroup','AveMonthSpend']].groupby('NChildrenAtHomeGroup').describe() )

aw_join=aw_join.join(aw_bb.set_index('CustomerID'), on='CustomerID', how='inner')
print( aw_join[['BikeBuyer','YearlyIncome']].groupby('BikeBuyer').median() )
print( aw_join[['BikeBuyer','NumberCarsOwned']].groupby('BikeBuyer').median() )
print( aw_join[['BikeBuyer','Occupation']].groupby('Occupation').count() )
print( aw_join[['BikeBuyer','Gender']].groupby('Gender').mean() )
print( aw_join[['BikeBuyer','MaritalStatus']].groupby('MaritalStatus').mean() )
print( aw_join[['CustomerID','Gender']].groupby('Gender').count() )
print( aw_join[['CustomerID','MaritalStatus']].groupby('MaritalStatus').count() )

aw_join.to_csv('AW_join.csv', index=False)

'''
def hist_plot(vals, lab):
   ## Distribution plot of Bike Buyer
   sns.distplot(vals)
   plt.title('Histogram of ' + lab)
   plt.xlabel('Value')
   plt.ylabel('Density')

# 
hist_plot(aw_bb['
'''
