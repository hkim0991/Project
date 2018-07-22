# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 12:15:11 2018

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


# Solution 1 - Numeric encoding -----------------------------------------------
# gather only categorical features for one-hot encoding process

category = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
            'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
            'Contract', 'PaperlessBilling', 'PaymentMethod']

cat_data = pd.DataFrame(data=train, columns=category)
cat_data.shape


from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

## Load a class : MultiColumnLabelEncoder 

class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)

# reference: https://stackoverflow.com/questions/24458645/label-encoding-across-multiple-columns-in-scikit-learn


# apply this class to cat_data 
        
encoded_cat = MultiColumnLabelEncoder(columns = category).fit_transform(cat_data)
encoded_cat.head()

for i in category:
    print("Frequency table for", i, "\n", encoded_cat[i].value_counts(), "\n")


# Back to Modeling - Decision Tree Classification -----------------------------
# Data combination: encoded categorical features + standardized continuous features

continuous = ['tenure', 'MonthlyCharges', 'TotalCharges']
cond_data = pd.DataFrame(data=train, columns= continuous)

final_data = pd.concat([cond_data, encoded_cat], axis=1)
final_data.info()


# Train/Test data partition ---------------------------------------------------

X_train = final_data
y_train = train['Churn']

X_tr, X_va, y_tr, y_va = train_test_split(X_train, y_train, test_size=0.3, random_state=0)
print(X_tr.shape, y_tr.shape) # 4930 x 19
print(X_va.shape, y_va.shape) # 2113 x 19


# Standarization of the continuous features

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_tr)
X_tr_scaled = scaler.transform(X_tr)
X_va_scaled = scaler.transform(X_va)


# Modeling - Decision Tree Classification -------------------------------------

tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_tr_scaled, y_tr)

print("train data accuracy: {:.3f}".format(tree.score(X_tr_scaled, y_tr))) # 0.997
print("test data accuracy: {:.3f}".format(tree.score(X_va_scaled, y_va)))  # 0.728 -> overfitting


# Decision Tree visualization -------------------------------------------------

 #!pip install graphviz 
 #!pip install pydotplus 
 # Let's not forget to add the path of graphviz to PATH in environment variables 
 # Then, restart your Python IDE

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
with open ('dtree_pipe1.png', 'wb') as f:
    f.write(png_bytes)
    
Image(png_bytes)

#graph = pydotplus.graph_from_dot_data(dot_data)
#Image(graph.create_png())


# Model tuning - Decision Tree Classification ---------------------------------
tree2 = DecisionTreeClassifier(max_depth= 10,  # original model : 25
                               max_leaf_nodes=50,
                               # max_features = 10,
                               min_samples_leaf = 3,
                               random_state=0)
tree2.fit(X_tr, y_tr)

print("train data accuracy: {:.3f}".format(tree2.score(X_tr, y_tr))) # 0.831
print("test data accuracy: {:.3f}".format(tree2.score(X_va, y_va)))  # 0.790


dot_data2 = export_graphviz(tree2, out_file=None, 
                           feature_names = data_feature_names,
                           class_names='Churn')

graph = Source(dot_data2)
png_bytes2 = graph.pipe(format='png')
with open ('dtree_pipe.png', 'wb') as f:
    f.write(png_bytes2)
    
Image(png_bytes2)


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
        for n in range(2, 6):
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
#0.7865593942262187 381

# Ploting the results ---------------------------------------------------------
fig = plt.figure(figsize=(12,8))
plt.plot(lni, "--", label="train set", color="blue")
plt.plot(lnr, "-", label="test set", color="red")
plt.plot(numMax, max, "o")
ann = plt.annotate("is" % str(n))
plt.legend()
plt.show()

#trial #: 380 
#max_depth:  7 | max_leaf_nodes:  23 | min_samples_leaf:  2
#train data accuracy: 0.810
#test data accuracy: 0.787


tree_final = DecisionTreeClassifier(max_depth=7,  
                               max_leaf_nodes= 23, 
                               min_samples_leaf = 2,
                               random_state=0)
tree_final.fit(X_tr_scaled, y_tr)

print("train data accuracy: {:.3f}".format(tree_final.score(X_tr_scaled, y_tr))) # 0.810
print("test data accuracy: {:.3f}".format(tree_final.score(X_va_scaled, y_va)))  # 0.787


# Decision Tree Model Visualization after tuning ------------------------------
dot_data3 = export_graphviz(tree_final, out_file=None, 
                           feature_names = data_feature_names,
                           class_names='Churn', filled=True)

graph = Source(dot_data3)
png_bytes = graph.pipe(format='png')
with open ('dtree_pipe_final.png', 'wb') as f:
    f.write(png_bytes)
    
Image(png_bytes)


# Feature importance ----------------------------------------------------------

fig = plt.figure(figsize=(10,8))
plot_feature_importances_telco(tree_final)

