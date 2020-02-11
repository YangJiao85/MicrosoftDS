## Construct model
## Labels
## Category: BikeBuyer
## Numeric:  AveMonthSpend --> log_AveMonthSpend
import pandas as pd
from sklearn import preprocessing
import sklearn.model_selection as ms
from sklearn import linear_model
import sklearn.metrics as sklm
import numpy as np
import numpy.random as nr
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
import math

# Import data for customer information
aw = pd.read_csv('AW_prep.csv')
print( 'Load AW_prep.csv')

cat_cols = ['Education', 'Occupation', 'Gender', 'MaritalStatus', 'HomeOwnerFlag', 'NCarsGroup', 'NChildrenAtHomeGroup', 'TotalChildrenGroup']
num_cols = ['YearlyIncome', 'Age']

## Prepare the model matrix
labels = np.array(aw['BikeBuyer'])
## Categorical features

def encode_string(cat_feature):
    ## First encode the strings to numeric categories
    enc = preprocessing.LabelEncoder()
    enc.fit(cat_feature)
    enc_cat_feature = enc.transform(cat_feature)
    ## Now, apply one hot encoding
    ohe = preprocessing.OneHotEncoder(categories='auto')
    encoded = ohe.fit(enc_cat_feature.reshape(-1,1))
    return encoded.transform(enc_cat_feature.reshape(-1,1)).toarray()

for col in cat_cols:
    temp = encode_string(aw[col])
    if (col == cat_cols[0]):
        Features = temp
    else:
        Features = np.concatenate([Features, temp], axis = 1)

print('Transformed categorical features')
print(Features.shape)
print(Features[:2, :])

## Numeric features
Features = np.concatenate([Features, np.array(aw[num_cols])], axis = 1)
print('Transformed features')
print(Features.shape)
print(Features[:2,:])

## Split the dataset
nr.seed(1234)
indx = range(Features.shape[0])
indx = ms.train_test_split(indx, test_size = 2000)
X_train = Features[indx[0],:]
y_train = np.ravel(labels[indx[0]])
X_test = Features[indx[1],:]
y_test = np.ravel(labels[indx[1]])

## Rescale numeric features
scaler = preprocessing.StandardScaler().fit(X_train[:,27:])
X_train[:,27:] = scaler.transform(X_train[:,27:])
X_test[:,27:] = scaler.transform(X_test[:,27:])
print(X_train.shape)
print(X_train[:5,:])

## Construct the logistic regression model
##   Define and fit the logistic regression model
logistic_mod = linear_model.LogisticRegression()
logistic_mod.fit(X_train, y_train)

print(logistic_mod.intercept_)
print(logistic_mod.coef_)

probabilities = logistic_mod.predict_proba(X_test)
print(probabilities[:15,:])

## Score and evaluate the classification model
def score_model(probs, threshold):
    return np.array([1 if x > threshold else 0 for x in probs[:,1]])
scores = score_model(probabilities, 0.5)
print(np.array(scores[:15]))
print(y_test[:15])

def print_metrics(labels, scores):
    metrics = sklm.precision_recall_fscore_support(labels, scores)
    conf = sklm.confusion_matrix(labels, scores)
    print('                 Confusion matrix')
    print('                 Score positive    Score negative')
    print('Actual positive    %6d' % conf[0,0] + '             %5d' % conf[0,1])
    print('Actual negative    %6d' % conf[1,0] + '             %5d' % conf[1,1])
    print('')
    print('Accuracy  %0.2f' % sklm.accuracy_score(labels, scores))
    print(' ')
    print('           Positive      Negative')
    print('Num case   %6d' % metrics[3][0] + '        %6d' % metrics[3][1])
    print('Precision  %6.2f' % metrics[0][0] + '        %6.2f' % metrics[0][1])
    print('Recall     %6.2f' % metrics[1][0] + '        %6.2f' % metrics[1][1])
    print('F1         %6.2f' % metrics[2][0] + '        %6.2f' % metrics[2][1])



print_metrics(y_test, scores)

def plot_auc(labels, probs):
    ## Compute the false positive rate, true positive rate
    ## and threshold along with the AUC
    fpr, tpr, threshold = sklm.roc_curve(labels, probs[:,1])
    auc = sklm.auc(fpr, tpr)

    ## Plot the result
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, color = 'orange', label = 'AUC = %0.2f' % auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

plot_auc(y_test, probabilities)

## A naive 'classifier'
probs_positive = np.concatenate((np.ones((probabilities.shape[0], 1)),
                                 np.zeros((probabilities.shape[0], 1))),
                                 axis = 1)
scores_positive = score_model(probs_positive, 0.5)
print_metrics(y_test, scores_positive)
plot_auc(y_test, probs_positive)

## Compute a weighted model
#logistic_mod = linear_model.LogisticRegression(class_weight = {0:0.45, 1:0.55})
#logistic_mod.fit(X_train, y_train)
#probabilities = logistic_mod.predict_proba(X_test)
#print(probabilities[:15,:])
#
#scores = score_model(probabilities, 0.5)
#print_metrics(y_test, scores)
#plot_auc(y_test, probabilities)

## Find a better shreshold
def test_threshold(probs, labels, threshold):
    scores = score_model(probs, threshold)
    print('')
    print('For threshold = ' + str(threshold))
    print_metrics(labels, scores)

thresholds = [0.5, 0.45, 0.40, 0.35, 0.3, 0.25]
for t in thresholds:
    test_threshold(probabilities, y_test, t)


######################################################################
## Predict new input features
aw_cust_test = pd.read_csv('AW_test.csv')
print(aw_cust_test.columns)
print(aw_cust_test.shape)

def trans_features(aw_custs):
    ## new column 'age' by data collected date 1st January 1998 - 'BirthDate'
    aw_custs['Age'] = (pd.to_datetime('1998-01-01') - pd.to_datetime(aw_custs['BirthDate'], errors='coerce')).astype('<m8[Y]')
    ##    Aggregating categorical variables
    ## AgeGroup <25 , 25-45, 45-55, >55
    aw_custs['AgeGroup'] = pd.cut(aw_custs['Age'], bins = [0,25,45,55,1000], \
              labels = ['<25','25-45','45-55','>55'], right=False)
    aw_custs['NCarsGroup'] = pd.cut(aw_custs['NumberCarsOwned'], \
              bins = [-1,0,2,10], labels = ['No','1-2','>=3'], right=True)
    aw_custs['NChildrenAtHomeGroup'] = pd.cut(aw_custs['NumberChildrenAtHome'], \
              bins = [-1,0,100], labels = ['No','>=1'], right=True)
    aw_custs['TotalChildrenGroup'] = pd.cut(aw_custs['TotalChildren'], \
              bins = [-1,0,1,2,3,4,100], labels = ['No','1','2','3','4','>=5'], right=True)

    for col in cat_cols:
        temp = encode_string(aw_custs[col])
        if (col == cat_cols[0]):
            Features = temp
        else:
            Features = np.concatenate([Features, temp], axis = 1)

    print('Transformed categorical features')
    print(Features.shape)
    print(Features[:2, :])

    ## Numeric features
    Features = np.concatenate([Features, np.array(aw_custs[num_cols])], axis = 1)
    Features[:,27:] = scaler.transform(Features[:,27:])
    print('Transformed features')
    print(Features.shape)
    print(Features[:2,:])
    return Features

X_new = trans_features(aw_cust_test)
probabilities = logistic_mod.predict_proba(X_new)
scores = score_model(probabilities, 0.5)
print(probabilities[:15,:])
print(type(scores))
print(len(scores))
print(type(np.array(aw_cust_test['CustomerID'])))
print(len(np.array(aw_cust_test['CustomerID'])))

aw_BikeBuyer_pred = pd.DataFrame({'CustomerID': np.array(aw_cust_test['CustomerID']),'BikeBuyer_pred': scores[:]})
aw_BikeBuyer_pred.to_csv('AW_BikeBuyer_pred.csv',index=False)

