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
train_path = "../data/telco/telco_data_preprocessed.csv"

train = pd.read_csv(train_path, engine='python')

train.shape # 7043 x 21
train.head()
train.info()
train.isnull().sum() # no missing value
train.describe()

# 20 predictor variables and 1 target variable('Churn')
train['Churn'].value_counts() # no:5174, yes:1869


# Modeling - Decision Tree Classification -------------------------------------

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

train2 = train.copy()

X_train = train2.drop('Churn',axis=1, inplace=True)
y_train = train['Churn']

X_tr, X_va, y_tr, y_va = train_test_split(X_train, y_train, test_size=0.3, random_state=0)
print(X_tr.shape, y_tr.shape)
print(X_va.shape, y_va.shape)

tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_tr, y_tr) 
# !! ISSUE !!: this model only consider numerical categorical featutres as categorical features.


# Solution 2 - One-hot encoding -----------------------------------------------
# gather only categorical features for one-hot encoding process

category = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
            'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
            'Contract', 'PaperlessBilling', 'PaymentMethod']

cat_data = pd.DataFrame(data=train, columns=category)

cat_data.shape
print('features from original data: \n', list(cat_data.columns))


# apply 'get_dummies' function to cat_data

data_dummies = pd.get_dummies(cat_data)

print('features after get_dummies: \n', list(data_dummies.columns))
data_dummies.head()


# Back to Modeling - Decision Tree Classification -----------------------------
# Data combination: encoded categorical features + continuous features

continuous = ['tenure', 'MonthlyCharges', 'TotalCharges']
cond_data = pd.DataFrame(data=train, columns= continuous)

final_data = pd.concat([cond_data, data_dummies], axis=1)
final_data.info()

# write the data after one-hot-encoding into a csv
# final_data.to_csv('telco_data_after_onehotencoding.csv', index=False)


# Train/Test data partition ---------------------------------------------------

X_train = final_data
y_train = train['Churn']

X_tr, X_va, y_tr, y_va = train_test_split(X_train, y_train, test_size=0.3, random_state=0)
print(X_tr.shape, y_tr.shape)
print(X_va.shape, y_va.shape)


# Standarization of the features

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_tr)
X_tr_scaled = scaler.transform(X_tr)
X_va_scaled = scaler.transform(X_va)


# Modeling - Decision Tree Classification -------------------------------------

tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_tr_scaled, y_tr)

print("train data accuracy: {:.3f}".format(tree.score(X_tr_scaled, y_tr))) # 0.997
print("test data accuracy: {:.3f}".format(tree.score(X_va_scaled, y_va)))  # 0.733 -> overfitting


# Decision Tree visualization -------------------------------------------------

from sklearn.tree import export_graphviz
from IPython.display import Image
from graphviz import Source
import pydotplus
import graphviz

data_feature_names = final_data.columns.values.tolist()

dot_data = export_graphviz(tree, out_file=None, 
                           feature_names= data_feature_names,
                           class_names='Churn')

graph = Source(dot_data)
png_bytes = graph.pipe(format='png')
with open ('dtree_pipe_onehot.png', 'wb') as f:
    f.write(png_bytes)
    
Image(png_bytes)

#graph = pydotplus.graph_from_dot_data(dot_data)
#Image(graph.create_png())


# Feature importance ----------------------------------------------------------

print("feature importance:\n{}".format(tree.feature_importances_)) 

def plot_feature_importances_telco(model):
    n_features = final_data.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), final_data.columns)
    plt.xlabel("feature importance")
    plt.ylabel('features')
    plt.ylim(-1, n_features)

fig = plt.figure(figsize=(10,8))
plot_feature_importances_telco(tree)


## find out the most highest accuracy for test dataset ------------------------
max=0; numMax= 0; cnt= 0
l1 = []
lni = []
lnr = []
for i in range(5, 21):
    for j in range(10, 51):
        for n in range(2, 6, 1):
            print("trial #:", cnt, "\n", "max_depth: ", i, "| max_leaf_nodes: ", j, "| min_samples_leaf: ", n)
            tree = DecisionTreeClassifier(max_depth = i,  # original model : 25
                               max_leaf_nodes = j,
                               min_samples_leaf = n,
                               random_state=0)
            tree.fit(X_tr_scaled, y_tr)
            treetest = tree.score(X_va_scaled, y_va)
            print("train data accuracy: {:.3f}".format(tree.score(X_tr_scaled, y_tr))) 
            print("test data accuracy: {:.3f}".format(tree.score(X_va_scaled, y_va)))
            lni.append(tree.score(X_tr_scaled, y_tr))
            lnr.append(tree.score(X_va_scaled, y_va))
            cnt += 1
            l1.append(cnt)
            if max < treetest:
                max = treetest
                numMax = cnt

print(max, numMax)
# 0.794131566493 69


# Ploting the results ---------------------------------------------------------
fig = plt.figure(figsize=(10,8))
plt.plot(lni, "--", label="train set", color="blue")
plt.plot(lnr, "-", label="test set", color="red")
plt.plot(numMax, max, "o")
ann = plt.annotate("is" % str(n))
plt.legend()
plt.show()


# Final model with tuning parameters ------------------------------------------

# trial #: 68 
# max_depth:  5 | max_leaf_nodes:  27 | min_samples_leaf:  2
# train data accuracy: 0.809
# test data accuracy: 0.794


tree_final = DecisionTreeClassifier(max_depth=5,  
                               max_leaf_nodes= 27, 
                               min_samples_leaf = 2,
                               random_state=0)
tree_final.fit(X_tr_scaled, y_tr)

print("train data accuracy: {:.3f}".format(tree_final.score(X_tr_scaled, y_tr))) # 0.809
print("test data accuracy: {:.3f}".format(tree_final.score(X_va_scaled, y_va)))  # 0.794


# Decision Tree Model Visualization after tuning ------------------------------

dot_data3 = export_graphviz(tree_final, out_file=None, 
                           feature_names = data_feature_names,
                           class_names='Churn', filled=True)

graph = Source(dot_data3)
png_bytes = graph.pipe(format='png')
with open ('dtree_pipe_onehot_final.png', 'wb') as f:
    f.write(png_bytes)
    
Image(png_bytes)


# Feature importance after tuning ---------------------------------------------

fig = plt.figure(figsize=(10,8))
plot_feature_importances_telco(tree_final)

