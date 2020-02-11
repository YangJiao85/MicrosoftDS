##
## Labels
## Numeric:   rate_spread
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import numpy.random as nr
import math

## Import data
hmda_train=pd.read_csv('hmda_train.csv')
cat_cols = ['msa_md', 'state_code', 'county_code', \
            'lender', 'loan_type', 'property_type', 'loan_purpose', 'occupancy', 'preapproval', \
            'applicant_ethnicity', 'applicant_race', 'applicant_sex', 'co_applicant'] 
num_cols = ['loan_amount', 'applicant_income', \
            'population', 'minority_population_pct', 'ffiecmedian_family_income', 'tract_to_msa_md_income_pct', 'number_of_owner_occupied_units', 'number_of_1_to_4_family_units']
##  Identity row_id
##  Label    rate_spread

print(hmda_train.head())
print(hmda_train.shape)
print(hmda_train.columns)
print(hmda_train.dtypes)



