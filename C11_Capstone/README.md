# Analysis of Home Mortgage Rate
**November 2019**

This analysis explore data concerning how demographics, location, property type, lender and other factors are related to the mortgage rate offered to applicants. The analysis is based on 200,000 observations from home mortgage disclosure act (HMDA) data, each containing specific characteristics of an loan application.

Potential relationships between characteristics and rate spread were identified by statistical analysis and data visualization. A model to predict this rate for loan applicants was created.

The author reached the following conclusions based on the analysis and modeling:

- The most affecting variables are loan information features, including property type, loan amount, lender and loan type. Specifically,
  - **Property types** Manufactured housing has higher rate spread than family housing.
  - **Lenders** Some lenders have higher rate spread.
  - **Loan types** FHA-insured loan has lower rate spread than conventional ones.
  - **Loan amount** High loan amount is related to low rate spread. Low loan amount corresponds to wide range of rate spread.
- An efficient prediction also rely on other loan information, applicant information, demographics and property location.

A [report](./report/MPP.pdf) presents the exploratory data analysis and a random forest regressor model that was used to predict the rate spread of a new dataset of 200,000 observations.

