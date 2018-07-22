# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 12:48:52 2018

@author: kimi
"""

# Import libraries & funtions -------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

print(os.getcwd())
#os.chdir('C:/Users/202-22/Documents/Python - Hyesu/Project/telco')
os.chdir('D:/Data/Python/project')


# Load dataset ----------------------------------------------------------------

train_path = "../data/telco/telco_train.csv"

train = pd.read_csv(train_path, engine='python')

train.shape # 7043 x 21
train.head()
train.info()
train.isnull().sum() # no missing value now*
train.describe()  # 3 numerical features

# 20 predictor variables and 1 target variable('Churn')
train['Churn'].value_counts() # no:5174, yes:1869


# Feature conversion ----------------------------------------------------------
# features 'SeniorCitizen' should be changed to categorical feature (0 or 1)
# feature 'TotalCharges' should be changed to continuous feature

## 1. continuous to categorical - SeniorCitizen, tenure*
for col in ['SeniorCitizen']:
    train[col] = train[col].astype('object')

train['SeniorCitizen'] = np.where(train['SeniorCitizen']==0, 'No', 'Yes')


## 2. categorical to continuous(float)- TotalCharges
train['TotalCharges'] = pd.to_numeric(train['TotalCharges'], errors='coerce')

train.info()
train.isnull().sum() # 11 missing values in TotalCharges feature after changing its data type from  object to float 
train.describe() # 3 numerical features: MonthlyChargees, TotalCharges + tenure


# EDA -------------------------------------------------------------------------
# Categorical features - 1. frequency table -----------------------------------

category = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
            'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
            'Contract', 'PaperlessBilling', 'PaymentMethod']

    
# For some features, we group some values which are considered as same meaning 
# for example, in the feature 'MultipleLines': 'No phone service' is considered as 'No'

cat_list = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
            'TechSupport', 'StreamingTV', 'StreamingMovies']

for j in cat_list:
    train[j] = np.where(train[j]=='Yes', 'Yes', 'No')


# Checking if there is any typing error in value.   

for i in category:
    print("Frequency table for", i, "\n", train[i].value_counts(), "\n")


# Categorical features - countplot --------------------------------------------
    
for i in category:
    plt.figure(figsize=(7,5))
    sns.countplot(train[i])
    plt.xticks(rotation=45) # Q. Only see the last feautre... WHY? A. plt.show()
    plt.show()


# categorical features related to demographics --------------------------------
    
demo = ['gender', 'SeniorCitizen', 'Partner', 'Dependents']

train['SeniorCitizen'] = np.where(train['SeniorCitizen']==0, 'No', 'Yes')

cnt=0
fig = plt.figure(figsize=(12,8))
fig.subplots_adjust(hspace=0.4, wspace=0.4)  # space between plots
for j in demo:
    cnt = cnt+1
    plt.subplot(2,2,cnt)
    sns.countplot(train[j], hue=train['Churn'])
    plt.title('group1 - demographics')
plt.show()


# Factorplot with Churn x demo theme features which is same as countplot with hun parameter
for j in demo:
    plt.figure(figsize=(10, 6))
    sns.factorplot(x='Churn', col=j, data=train, kind='count')
    sns.factorplot(x=j, col='Churn', data=train, kind='count')
plt.show()
    

# Categorical features related to service -------------------------------------

plt.figure(figsize=(10, 5)).subplots_adjust(wspace=0.3)
plt.subplot(121)
sns.countplot(train['PhoneService'], hue=train['Churn'])
plt.title('Phonce Service - yes or no')

plt.subplot(122)
sns.countplot(train['InternetService'], hue=train['Churn'])
plt.title('Internet Service type')


# Categorical feature related to payment --------------------------------------

plt.figure(figsize=(10, 8)).subplots_adjust(hspace=0.3)
plt.subplot(211)
sns.countplot(train['PaperlessBilling'], hue=train['Churn'])

plt.subplot(212)
sns.countplot(train['PaymentMethod'], hue=train['Churn'])
plt.xticks(rotation=45)


sns.factorplot(x='PaymentMethod', col='PaperlessBilling', data=train, kind='count')
plt.xticks(rotation=45)


# Categorical features related to contract ------------------------------------

plt.figure(figsize=(10, 10)).subplots_adjust(hspace=0.3)
plt.subplot(211)
sns.countplot(train['tenure'], hue=train['Churn'])  
plt.xticks(rotation=90)

plt.subplot(212)
sns.countplot(train['Contract'], hue=train['Churn'])


# Categorical features related to customer's interest -------------------------

plt.figure(figsize=(10, 5)).subplots_adjust(wspace=0.3)
plt.subplot(121)
sns.countplot(train['StreamingTV'], hue=train['Churn'])
plt.title('Streaming TV - yes or no')

plt.subplot(122)
sns.countplot(train['StreamingMovies'], hue=train['Churn'])
plt.title('Streaming Movice - yes or no')


# Others ----------------------------------------------------------------------

plt.figure(figsize=(10, 6)).subplots_adjust(hspace=0.4, wspace=0.3)
plt.subplot(221)
sns.countplot(train['OnlineSecurity'], hue=train['Churn'])

plt.subplot(222)
sns.countplot(train['OnlineBackup'], hue=train['Churn'])

plt.subplot(223)
sns.countplot(train['DeviceProtection'], hue=train['Churn'])

plt.subplot(224)
sns.countplot(train['TechSupport'], hue=train['Churn'])


# Target feature - Churn ------------------------------------------------------

sns.countplot(train['Churn'])

min(train['tenure'])
max(train['tenure'])


# Univariate Analysis ---------------------------------------------------------
# continuous variables - 1. Mean/std/min/max/IQR ------------------------------

train.describe()
train.isnull().sum() # 11 missing values in Total Charges (their tenure is 0)

# Normally we replace the missing value with the mean value
np.mean(train['TotalCharges']) # mean: 2283.300
min(train['TotalCharges']) # min: 18.80

# missing value treatment 
nan_rows = train[train['TotalCharges'].isnull()]

# Since their tenure is 0, the missing value should be replaced by 0 
train['TotalCharges'].fillna(0, inplace=True)
train['TotalCharges'].isnull().sum()

train.isnull().sum() # no missing values
train.info() # minmum value of 'tenure'has been changed to 0
train.describe()


# continous variables - 2. Histogram ------------------------------------------
cont = ['MonthlyCharges', 'TotalCharges', 'tenure']

# Basic histogram 
cnt=0
plt.figure(figsize=(10, 10)).subplots_adjust(hspace=0.4)
for i in cont:
    cnt = cnt +1
    plt.subplot(3, 1, cnt)
    plt.hist(train[i], bins=30)
    plt.title(i)
plt.show()


# Seaborn dist() plot
cnt=0
plt.figure(figsize=(10, 10)).subplots_adjust(hspace=0.3)
for i in cont:
    cnt = cnt +1
    plt.subplot(3, 1, cnt)
    sns.distplot(train[i], kde=True, rug=True)
plt.show()


plt.figure(figsize=(10, 8))
sns.distplot(train['MonthlyCharges'], kde=True, rug=True)

plt.figure(figsize=(10, 8))
sns.distplot(train['TotalCharges'], kde=True, rug=True)

sns.distplot(train['tenure'], kde=True, rug-True)


# Continous variables - 3. Boxplot --------------------------------------------

plt.figure(figsize=(16, 6)).subplots_adjust(hspace=0.4, wspace=0.3)
my_pal = {"No": "r", "Yes": "g"}

plt.subplot(131)
sns.boxplot(x=train['Churn'], y=train['MonthlyCharges'], palette=my_pal) 

plt.subplot(132)
sns.boxplot(x=train['Churn'], y=train['TotalCharges'], palette=my_pal) # outlier

plt.subplot(133)
sns.boxplot(x=train['Churn'], y=train['tenure'], palette=my_pal) # outlier


# There are some outliers in both features 'TotalCharges' and 'tenure' only when 'Churn' is Yes
# Should we do something?
# Let's find the border of outlier

churn_yes = pd.DataFrame(train.loc[train['Churn']=='Yes'])
churn_yes.describe()

quantile_cont = pd.DataFrame(churn_yes.describe(), columns=cont)

# outliers for 'TotalCharges' 
TotalCharges_upper_Q = quantile_cont['TotalCharges'][6]
TotalCharges_lower_Q = quantile_cont['TotalCharges'][4]

TotalCharges_IQR = TotalCharges_upper_Q - TotalCharges_lower_Q

TotalCharges_border = TotalCharges_upper_Q + (TotalCharges_IQR * 1.5) # 5626.5

TotalCharges_outliers = pd.DataFrame(churn_yes.loc[churn_yes['TotalCharges']> TotalCharges_border]) #109 outliers


# outliers for 'tenure'
tenure_upper_Q = quantile_cont['tenure'][6]
tenure_lower_Q = quantile_cont['tenure'][4]

tenure_IQR = tenure_upper_Q - tenure_lower_Q

tenure_border = tenure_upper_Q + (tenure_IQR * 1.5) #69.5

tenure_outliers = pd.DataFrame(churn_yes.loc[churn_yes['tenure']> tenure_border]) #23 outliers


# Continous variables - 4. Violinplot -----------------------------------------

plt.figure(figsize=(16,6)).subplots_adjust(hspace=0.4, wspace=0.3)
my_pal = {"No": "r", "Yes": "g"}

plt.subplot(131)
sns.violinplot(x=train['Churn'], y=train['MonthlyCharges'], palette=my_pal)

plt.subplot(132)
sns.violinplot(x=train['Churn'], y=train['TotalCharges'], palette=my_pal)

plt.subplot(133)
sns.violinplot(x=train['Churn'], y=train['tenure'], palette=my_pal)


# Bi-variate Analysis ---------------------------------------------------------
# continous x continous - 1. Correlation --------------------------------------

train['MonthlyCharges'].corr(train['TotalCharges']) # 0.65

train['MonthlyCharges'].corr(train['tenure']) # 0.25

train['TotalCharges'].corr(train['tenure']) #0.83


cont = ['tenure', 'MonthlyCharges', 'TotalCharges']

cont_cols = pd.DataFrame(data=train, columns=cont)
cont_cols.head()


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(cont_cols)
cont_scaled = scaler.transform(cont_cols)

corr = cont_cols.corr()
sns.heatmap(corr, xticklabels= corr.columns, yticklabels= corr.columns)


# continous x continous - 2. Scatter plot -------------------------------------

sns.jointplot(x='MonthlyCharges', y='TotalCharges', data=train) # correlation: 0.65 

sns.jointplot(x='tenure', y='MonthlyCharges', data=train) # correlation: 0.25

sns.jointplot(x='tenure', y='TotalCharges', data=train) # correlation: 0.83
 

sns.pairplot(train, size=3, vars=['MonthlyCharges', 'TotalCharges', 'tenure'], hue="Churn", markers=["o", "^"])
plt.show()

#train['tenure'] = train['tenure'].astype('object')
#train.describe()


# Categorical x categorical - 1. Two-way table --------------------------------
# Method 1
churn_sex = pd.crosstab(index=train['Churn'],
                        columns=train['gender'])
churn_sex.index= ['No', 'Yes']
churn_sex


# Method 2
churn_sex2 = train.groupby(['Churn', 'gender'])
churn_sex2.size()


# the result of table 1 is simpler, therefore i'm going to use the first method.

for i in category:
    cat_cross = pd.crosstab(index=train['Churn'],
                            columns=train[i])
    cat_cross.index = ['No', 'Yes']
    print("Two-way tables with Churn and", i, "\n", cat_cross, "\n")
    

# Categorical x categorical - 2. Stacked Column Chart -------------------------
    
for i in category:
    table = pd.crosstab(index=train['Churn'],
                        columns=train[i])
    table.plot(kind='bar', figsize=(10,8), stacked=True)
    plt.show()
 

# Categorical x continous - 1. Boxplot ----------------------------------------
    
plt.figure(figsize=(12.5,7.5))
sns.boxplot(x='gender', y='MonthlyCharges', hue='Churn', data=train, palette='Set2')

plt.figure(figsize=(12.5,7.5))
sns.boxplot(x='SeniorCitizen', y='MonthlyCharges', hue='Churn', data=train, palette='Set2')

plt.figure(figsize=(12.5,7.5))
sns.boxplot(x='Partner', y='MonthlyCharges', hue='Churn', data=train, palette='Set2')

plt.figure(figsize=(12.5,7.5))
sns.boxplot(x='Dependents', y='MonthlyCharges', hue='Churn', data=train, palette='Set2')


# Categorical x continous - 2. Scatterplots -----------------------------------

sns.stripplot(x='Contract', y='TotalCharges', data=train, jitter=True)
sns.stripplot(x='Contract', y='MonthlyCharges', data=train, jitter=True)


sns.swarmplot(x='Contract', y='TotalCharges', data=train)
sns.swarmplot(x='Contract', y='TotalCharges', hue='Churn', data=train)


## writing preprocessed dataset into a csv file -------------------------------

import csv
train_after = train.copy()
train_after.describe()
train_after.info()

train_after.to_csv('telco_data_preprocessed.csv', index=False)
