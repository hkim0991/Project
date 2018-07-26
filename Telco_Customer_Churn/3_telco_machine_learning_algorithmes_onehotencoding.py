# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 02:09:13 2018

@author: kimi
"""

# Import libraries & funtions ------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

print(os.getcwd())
#os.chdir('C:/Users/202-22/Documents/Python - Hyesu/Project/telco')
os.chdir('D:/Data/Python/project')


# Load dataset ----------------------------------------------------------------

# data after one-hot-encoding
train_path = "../data/telco/telco_data_after_onehotencoding.csv"
train = pd.read_csv(train_path, engine='python')

train.shape # 7043 x 39
train.head()
train.isnull().sum() # no missing value

# Load original dataset 
data_path = "../data/telco/telco_data_preprocessed.csv"
original = pd.read_csv(data_path, engine='python')

# for some ML models who doesn't convert string to float
# we change the data type of 'Churn' from string to integer
original['Churn'] = np.where(original['Churn']=='No', 0, 1) 
original.shape # 7043 x 21
original.head()


# Train/Test data partition ---------------------------------------------------

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

final_data = train.copy()
X_train = final_data
y_train = original['Churn']

X_tr, X_va, y_tr, y_va = train_test_split(X_train, y_train, test_size=0.3, random_state=0)
print(X_tr.shape, y_tr.shape)
print(X_va.shape, y_va.shape)


# Standarization of the features ----------------------------------------------

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_tr)
X_tr_scaled = scaler.transform(X_tr)
X_va_scaled = scaler.transform(X_va)


# Feature importance ----------------------------------------------------------

def plot_feature_importances_telco(model):
    n_features = final_data.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), final_data.columns)
    plt.xlabel("feature importance")
    plt.ylabel('features')
    plt.ylim(-1, n_features)



# Modeling - Logistic Regression ----------------------------------------------

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_tr_scaled, y_tr)

print("train data accuracy: {:.3f}".format(model.score(X_tr_scaled, y_tr))) # 0.806
print("test data accuracy: {:.3f}".format(model.score(X_va_scaled, y_va)))  # 0.802 


# Logistic Regression - Ridge Regulation (L2)
 
from sklearn.linear_model import Ridge

ridge = Ridge(alpha=10).fit(X_tr_scaled, y_tr)
 
print("train data accuracy: {:.3f}".format(ridge.score(X_tr_scaled, y_tr))) # 0.294
print("test data accuracy: {:.3f}".format(ridge.score(X_va_scaled, y_va)))  # 0.257 ???


# Logistic Regressing - Lasso Regulation (L1)  

from sklearn.linear_model import Lasso

lasso = Lasso().fit(X_tr_scaled, y_tr) 

print("train data accuracy: {:.3f}".format(lasso.score(X_tr_scaled, y_tr))) # 0.000
print("test data accuracy: {:.3f}".format(lasso.score(X_va_scaled, y_va)))  # -0.000 ???



# Modeling - knn --------------------------------------------------------------

from sklearn.neighbors import KNeighborsClassifier

training_accuracy = []
test_accuracy = []

neighbors_settings = range(3, 100)
for n in neighbors_settings:
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(X_tr_scaled, y_tr)
    training_accuracy.append(knn.score(X_tr_scaled, y_tr))
    test_accuracy.append(knn.score(X_va_scaled, y_va))
    
plt.plot(neighbors_settings, training_accuracy, label='Train accuracy')
plt.plot(neighbors_settings, test_accuracy, label='Test accuracy')
plt.ylabel('Accuracy')
plt.xlabel('n_neighbors')
plt.legend()
    
test_accuracy.index(max(test_accuracy)) # 94

knn = KNeighborsClassifier(n_neighbors=94)
knn.fit(X_tr_scaled, y_tr)

print("train data accuracy: {:.3f}".format(knn.score(X_tr_scaled, y_tr))) # 0.794
print("test data accuracy: {:.3f}".format(knn.score(X_va_scaled, y_va)))  # 0.790



# Modeling - Random Forest ----------------------------------------------------

from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators=10000, max_features=6, random_state=1) 
# for the classification: max_features=sqrt(n_features)

forest.fit(X_tr_scaled, y_tr) 

print("train data accuracy: {:.3f}".format(forest.score(X_tr_scaled, y_tr))) # 0.997 - overfitting
print("test data accuracy: {:.3f}".format(forest.score(X_va_scaled, y_va)))  # 0.787

fig = plt.figure(figsize=(10,8))
plot_feature_importances_telco(forest)


# to find the best parameters 
for i in range(5, 11):
    forest = RandomForestClassifier(n_estimators=10000, max_features=i, max_depth=5, random_state=1)
    forest.fit(X_tr_scaled, y_tr)
    print("train data accuracy: {:.3f}".format(forest.score(X_tr_scaled, y_tr)))
    print("test data accuracy: {:.3f}".format(forest.score(X_va_scaled, y_va))) 


# Final model with tuning parameters 

forest_final= RandomForestClassifier(n_estimators=10000, max_features=8, max_depth=5, random_state=1) 
forest_final.fit(X_tr_scaled, y_tr) 

print("train data accuracy: {:.3f}".format(forest_final.fit.score(X_tr_scaled, y_tr))) # 0.808
print("test data accuracy: {:.3f}".format(forest_final.fit.score(X_va_scaled, y_va)))  # 0.798

fig = plt.figure(figsize=(10,8))
plot_feature_importances_telco(forest_final)
    


# Modeling - Gradient Boosting Classifier -------------------------------------

from sklearn.ensemble import GradientBoostingClassifier

gbrt = GradientBoostingClassifier(n_estimators=10000, random_state=0)
gbrt.fit(X_tr_scaled, y_tr)

print("train data accuracy: {:.3f}".format(gbrt.score(X_tr_scaled, y_tr))) # 0.833
print("test data accuracy: {:.3f}".format(gbrt.score(X_va_scaled, y_va)))  # 0.799
# even without tuning, the performance is pretty fine. 


## find out the most highest accuracy for test dataset 

max=0; numMax= 0; cnt= 0
l1 = []
lni = []
lnr = []

list = [0.001, 0.003, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5]
for i in range(100, 1000, 100):
    for j in range(1, 6):
        for k in list:
            print('trial #:', cnt, '\n', "n_estimators: ", i, "| max_depth: ", j, "| learning rate: ", k)
            gbrt = GradientBoostingClassifier(n_estimators= i, 
                                              max_depth= j, 
                                              learning_rate= k, 
                                              random_state=0)
            gbrt.fit(X_tr_scaled, y_tr)
            treetest = gbrt.score(X_va_scaled, y_va)
            print("train data accuracy: {:.3f}".format(gbrt.score(X_tr_scaled, y_tr))) 
            print("test data accuracy: {:.3f}".format(gbrt.score(X_va_scaled, y_va))) 
            lni.append(gbrt.score(X_tr_scaled, y_tr))
            lnr.append(gbrt.score(X_va_scaled, y_va))
            cnt += 1
            l1.append(cnt)
            if max < treetest:
                max = treetest
                numMax = cnt

print(max, numMax)
#0.8078561287269286 150

# trial #: 149 
# n_estimators:  700 | max_depth:  1 | learning rate:  0.3
# train data accuracy: 0.818
# test data accuracy: 0.808


# Final model with tuning parameters 
 
gbrt_final = GradientBoostingClassifier(n_estimators=700, max_depth=1, learning_rate=0.3, random_state=0)
gbrt_final.fit(X_tr_scaled, y_tr)

print("train data accuracy: {:.3f}".format(gbrt_final.score(X_tr_scaled, y_tr))) # 0.818
print("test data accuracy: {:.3f}".format(gbrt_final.score(X_va_scaled, y_va)))  # 0.808

fig = plt.figure(figsize=(10,8))
plot_feature_importances_telco(gbrt_final) # almost only used continuous features (tenure, monthlycharges, and totalCharges)
    








