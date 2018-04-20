# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 15:56:45 2018

@author: Hyesu KIM
"""

## Data exploiration
# 01. Reading data
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# medi = pd.read_csv('C:\\Users\\Hyesu KIM\\Documents\\Python_Hyesu\\Python_project\\Medical\\medical_2014.csv', engine='python')

medi = pd.read_csv('C:\\Users\\202-22\\Documents\\Python - Hyesu\\Project\\Medical\\medical_2014.csv', engine='python')
medi.head()


# 02.Data preprocessing

medi.shape  ## 10,000,000 rows x 31 columns 

medi.index
medi.columns

'''
Original columns names in korean: 
'기준년도', '가입자일련번호', '성별코드', '연령대코드(5세단위)', '시도코드', '신장(5Cm단위)',
'체중(5Kg 단위)', '허리둘레', '시력(좌)', '시력(우)', '청력(좌)', '청력(우)', '수축기혈압',
'이완기혈압', '식전혈당(공복혈당)', '총콜레스테롤', '트리글리세라이드', 'HDL콜레스테롤', 'LDL콜레스테롤',
'혈색소', '요단백', '혈청크레아티닌', '(혈청지오티)AST', '(혈청지오티)ALT', '감마지티피', '흡연상태',
'음주여부', '구강검진 수검여부', '치아우식증유무', '치석유무', '데이터 기준일자'
'''
       
medi1 = medi.rename(index=str, columns={'기준년도': 'year', '가입자일련번호': 'member_id', '성별코드': 'sex', 
                               '연령대코드(5세단위)': 'age_group', '시도코드': 'city_code', 
                               '신장(5Cm단위)': 'height', '체중(5Kg 단위)': 'weight', '허리둘레':'waist',
                               '시력(좌)': 'sight_left', '시력(우)': 'sight_right', '청력(좌)': 'hear_left', '청력(우)':'hear_right',
                               '수축기혈압':'bp_high', '이완기혈압': 'bp_low', '식전혈당(공복혈당)': 'blds',
                               '총콜레스테롤':'total_chole', '트리글리세라이드':'triglyceride', 'HDL콜레스테롤':'hdl_chole',
                               'LDL콜레스테롤':'ldl_chole', '혈색소':'hmg', '요단백':'olig_prote_cd', 
                               '혈청크레아티닌':'creatinine', '(혈청지오티)AST':'sgot_ast', '(혈청지오티)ALT':'sgpt_alt',
                               '감마지티피':'gamma_gtp', '흡연상태':'smoke', '음주여부':'drink_yn', 
                               '구강검진 수검여부':'dent_exam_yn', '치아우식증유무':'crs_yn', '치석유무':'ttr_yn', '데이터 기준일자':'data_date'})

medi1.head(n=10)
medi1.dtypes # to check data type: str() in R
medi1.describe() # to summarize the data: summary() in R
"""
Categorical variables:
    
- sex (male:1, female:2)
- age_group (total: 14 groups, only from 5-18 group)  # need to see the data visualisation to know why the max is 18
- city_code (total: 17 groups, 41(Gyeonggi) > 11(Seoul) > 26(Busan) >48(Gyeongsangnamdo) > 47(GyeongSangBukdo))
- height (grouped by 5cm, 130~195cm, max:195)
- weight (30~140, min: 30kg ??? )  ## to check
- waist (51~129, max: 129cm ????)  ## to check
- sight_left (0.1~2.5, sight under 0.1 = 0.1, blindness:9.9)
- sight_right (0.1~2.5, sight under 0.1 = 0.1, blindness:9.9)

- smoke (non-smoking:1, quitted:2, smoke:3)
- drink_yn (no:0, yes:1)
- dental examination (no:0, yes:1)
- crs_yn (no:0, yes:1)
- ttr_yn (no:0, yes:1)

Continuous variables:
- blds (70 < normal < 100 , min=28, max=780 ???) ## hypoglycemia < 70, 100< dangerous < 126,  diabetes >= 126
- creatinine (normal: 0.8~1.7, 0.1~98, max=98 ????) ## to check
- sgot_ast (normal: 0~40 IU/L, 1~999, max=999 ???)  ## to check
- sgpt_alt (normal: 0~40 IU/L, 1~999, max=999 ???)  ## to check 
- gamma_gtp (normal: male - 11~63IU/L, female- 8~35IU/Lm, 1~999, max= 999 ???)
"""





## make a new column derived from existing columns
## BMI = Weight/height**2
'''
BMI = weight/height**2
obesity level
BMI < 18.5 == "Underweight"
BMI < 25 == "Normal"
BMI < 30 == "Overweight"
BMI >= 30 == "Obesity"
'''
medi1['BMI'] = medi1['weight']/(medi1['height']/100)**2  ## /100: because BMI unit is m
medi1['BMI']

medi1['Obesity'] = np.nan
medi1['Obesity'].loc[medi1['BMI'] < 18.5] = "Underweight"
medi1['Obesity'].loc[(medi1['BMI'] >= 18.5) & (medi1['BMI'] < 25)] = "Normal"
medi1['Obesity'].loc[(medi1['BMI'] >= 25) & (medi1['BMI'] < 30)] = "Overweight"
medi1['Obesity'].loc[medi1['BMI'] >= 30] = "Obesity"


medi1['Obesity']
     




## continuous variables to categorial variables
# df['col_name'] = df['col_name'].astype(object)
'''
def to_category(variable_name):
    list = ['sex', 'age_group', 'city_code', 'olig_prote_cd', 'smoke', 'drink_yn', 'dent_exam_yn', 'crs_yn','ttr_yn', 'Obesity']
    if variable_name in list:
        medi1[variable_name] = medi1[variable_name].astype('category')
'''


list = ['sex', 'age_group', 'city_code', 'olig_prote_cd', 'smoke', 'drink_yn', 'dent_exam_yn', 'crs_yn','ttr_yn', 'Obesity']
for i in list:
    medi1[i] = medi1[i].astype('category')
    print("the type of", i, "is", medi1[i].dtypes)

'''
medi1['sex'] = medi1['sex'].astype('category')
medi1['age_group'] = medi1['age_group'].astype('category')
medi1['city_code'] = medi1['city_code'].astype('category')
medi1['olig_pr'] = medi1['olig_prote_cd'].astype('category')
medi1['smoke'] = medi1['smoke'].astype('category')
medi1['drink_yn'] = medi1['drink_yn'].astype('category')
medi1['dent_exam_yn'] = medi1['dent_exam_yn'].astype('category')
medi1['crs_yn'] = medi1['crs_yn'].astype('category')
medi1['ttr_yn'] = medi1['ttr_yn'].astype('category')
'''

medi1['age_group'].describe()   ## 14 categories out of 18, top category is 9(60~64 years old)
medi1['city_code'].describe()   ## 17 city codes out of 17, top category is 41(GyeongGi-do)
medi1['smoke'].describe()       ## top category is 1 (non-smoking)
medi1['drink_yn'].describe()    ## yes > no
medi1['dent_exam_yn'].describe() ## yes < no
medi1['crs_yn'].describe()    ## When summing data, NA (missing) values will be treated as zero
medi1['ttr_yn'].describe()    ## When summing data, NA (missing) values will be treated as zero
medi1['Obesity'].describe()   ## category 'normal' is the most frequent group in this variable
medi1['blds'].describe()





## find out if there is missing value and where
medi1.info()   ## count non-null cells in each column

medi1.isnull().any()  ## to find out if there is NaNs (True or False) 
medi1.isnull().sum()  ## to count how many NaNs   





## Question is how to handlie the missing values in categorical variables?

drink_nan = medi1[medi1['drink_yn'].isnull()]  # to show where is the missing value
print(drink_nan)

smoke_nan = medi1[medi1['smoke'].isnull()]
olig_nan = medi1[medi1['olig_prote_cd'].isnull()]
ldlc_nan = medi1[medi1['ldl_chole'].isnull()]

sight_left_nan = medi1[medi1['sight_left'].isnull()]





## to count the numbers in each category value
age_freq = medi1['age_group'].value_counts()
age_freq

'''
wierd to have the data in the group of 15-18
9     141696
11    139237
10    115046
12    103095
7      93322
13     89764
8      86420
6      66955
15     51556
14     51438
5      24172
16     21937
17     12415
18      2947
'''

city_freq = medi1['city_code'].value_counts()
city_freq


medi1['Obesity'].value_counts()
'''
Normal         633464
Overweight     295142
Obesity         38070
Underweight     33324
'''
medi1['Obesity'].mode()






#03. data exploration:plot, histogram, barplot, etc. 
## histogram of continous variables
'''
Continuous variables:
- height (grouped by 5cm, 130~195cm, max:195)
- weight (30~140, min: 30kg ??? )  ## 30: 15, 35: 486, 140: 4 
- waist (51~129, max: 129cm ????)  ## to check
- sight_left (0.1~2.5, sight under 0.1 = 0.1, blindness:9.9)
- sight_right (0.1~2.5, sight under 0.1 = 0.1, blindness:9.9)
- creatinine (normal: 0.8~1.7, 0.1~98, max=98 ????) ## to check
- sgot_ast (normal: 0~40 IU/L, 1~999, max=999 ???)  ## to check
- sgpt_alt (normal: 0~40 IU/L, 1~999, max=999 ???)  ## to check 
- gamma_gtp (normal: male - 11~63IU/L, female- 8~35IU/Lm, 1~999, max= 999 ???)
'''
medi1['height'].plot(kind='hist', bins=100) ## height is not continous variable
plt.xlabel('Height distribution')

height_freq = medi1['height'].value_counts()
height_freq


medi1['weight'].plot(kind='hist', bins=100)
plt.xlabel('Weight distribution')

weight_freq = medi1['weight'].value_counts()
weight_freq


medi1['waist'].plot(kind='hist', bins=100)
plt.xlabel('Waist distribution')

waist_freq = medi1['waist'].value_counts()
waist_freq



medi1['sight_left'].plot(kind='hist', bins=100)
plt.xlim(xmin = 0, xmax = 10)
plt.xlabel('sight_left distribution')

sight_left_freq = medi1['sight_left'].value_counts()  ## 9.9: 3455
sight_left_freq

## >>> from the graphs, we now treat [height, weight, waist, sight_left & sight_right] variables as categorical data 

'''
'수축기혈압':'bp_high', '이완기혈압': 'bp_low', '식전혈당(공복혈당)': 'blds',
'총콜레스테롤':'total_chole', '트리글리세라이드':'triglyceride', 'HDL콜레스테롤':'hdl_chole',
'LDL콜레스테롤':'ldl_chole', '혈색소':'hmg', '요단백':'olig_pr', 
'혈청크레아티닌':'creatinine', '(혈청지오티)AST':'sgot_ast', '(혈청지오티)ALT':'sgpt_alt',
'감마지티피':'gamma_gtp'
'''
                               

medi1['blds'].plot(kind='hist', bins=100) 
plt.xlabel('Fasting blood sugar')

b_freq = medi1['blds'].value_counts()
b_freq



medi1['total_chole'].plot(kind='hist', bins=100) ## normal: 150~250
plt.xlabel('Total_Cholestrol')


medi1['bp_high'].plot(kind='hist', bins=100)
plt.xlim(xmin = 50, xmax = 200)
plt.xlabel('Highest Blood Pressure')

medi1['bp_low'].plot(kind='hist', bins=100)
plt.xlim(xmin = 30, xmax = 130)
plt.xlabel('Lowest Blood Pressure')



## boxplot of continous variables

plt.boxplot(medi1['total_chole'])
plt.boxplot(medi1['blds'])

plt.boxplot(medi1['bp_high'])
plt.boxplot(medi1['bp_low'])


sns.boxplot(medi1['weight'])
sns.boxplot(medi1['height'])
sns.boxplot(medi1['waist'])

sns.boxplot(medi1['total_chole'])
sns.boxplot(medi1['hdl_chole'])
sns.boxplot(medi1['ldl_chole'])
sns.boxplot(medi1['blds'])
 

sns.boxplot(medi1['olig_prote_cd'])

sns.boxplot(medi1['creatinine'])
sns.boxplot(medi1['sgot_ast'])
sns.boxplot(medi1['sgpt_alt'])
sns.boxplot(medi1['gamma_gtp'])
sns.boxplot(medi1['BMI'])



# barplot for categorical variables
plt.plot(medi1['creatinine'])

sns.countplot(medi1['smoke'])
plt.xlabel('Smoking status', fontsize=10)
plt.ylabel('Numbers', fontsize=10)
plt.title('Smoking Status in 2014')


sns.countplot(medi1['sex'])
plt.xlabel('Sex', fontsize=10)
plt.ylabel('Numbers', fontsize=10)
plt.title('Gender distribution in 2014')


sns.countplot(medi1['age_group'])
plt.xlabel('Age Group', fontsize=10)
plt.ylabel('Numbers', fontsize=10)
plt.title('Age Group in 2014')


sns.countplot(medi1['city_code'])
plt.xlabel('City Code', fontsize=10)
plt.ylabel('Numbers', fontsize=10)
plt.title('City Code in 2014')


sns.countplot(medi1['olig_prote_cd'])
plt.xlabel('olig_prote_cd', fontsize=10)
plt.ylabel('Numbers', fontsize=10)
plt.title('protein in urine  in 2014')


sns.countplot(medi1['drink_yn'])
plt.xlabel('Drinking status', fontsize=10)
plt.ylabel('Numbers', fontsize=10)
plt.title('Drinking Status in 2014')

sns.countplot(medi1['Obesity'])
plt.xlabel('Obesity status', fontsize=10)
plt.ylabel('Numbers', fontsize=10)
plt.title('Obesity Status in 2014')


## draw plotscatter between two variables
def pltscatter(x, y): 
    plt.scatter(x, y, c = "g", alpha = 0.2, label = "")
    plt.xlabel(x.name)
    plt.ylabel(y.name)
    plt.legend(loc = 2)
    plt.show()

pltscatter(medi1["hdl_chole"], medi1["ldl_chole"])
pltscatter(medi1["total_chole"], medi1["ldl_chole"])




## 04. Missing data handling

## drop the missing values for sight left/right, hear left/right

medi1 = medi1.loc[medi1['sight_left'].notnull()]  ##  select only the data which is not null.
medi1 = medi1.loc[medi1['sight_right'].notnull()]
medi1 = medi1.loc[medi1['hear_left'].notnull()]
medi1 = medi1.loc[medi1['hear_right'].notnull()]

medi1.isnull().sum()
medi1.info()

## fill the data with the most frequent data for categorical variables
# smoke
most_freq = medi1['smoke'].mode()
medi1['smoke'].fillna(most_freq[0], inplace=True)

# drink_yn
most_freq1 = medi1['drink_yn'].mode()
medi1['drink_yn'].fillna(most_freq1[0], inplace=True)


medi1.isnull().sum()

'''
## to fill the missing value with the most frequent value (categorical)
freq = tdf['Embarked'].value_counts() ## to count the value in each category
most_freq =  tdf['Embarked'].mode()  ## to find out the most frequent category
tdf2['Embarked'].fillna(most_freq[0], inplace=True)
'''




## choose only useful variables: missing only 252 values : we drop 'crs_yn' and 'ttr_yn'
'olig_prote_cd'

c = ['member_id', 'sex', 'age_group', 'city_code', 'height', 'weight', 'waist', 
     'sight_left', 'sight_right', 'hear_left', 'hear_right', 
     'bp_high', 'bp_low', 'blds', 'total_chole', 'triglyceride',
     'hdl_chole', 'ldl_chole', 'hmg', 'creatinine', 
     'sgot_ast', 'sgpt_alt', 'gamma_gtp', 'smoke', 'drink_yn', 'dent_exam_yn','BMI', 'Obesity']
medi2 = medi1[c]
medi2.info()
medi2.isnull().sum()


sns.pairplot(medi2)  

## 05. Data mininig: predict missing values in ldl_chole with regression analysis 
## step1. make a seperate dataset with/without the missing value : train data and test data
## step2. drop the missing data to make a train/validation dataset

## df['name of variable'].dropna(axis=0 -> row, axis=1 => column)

## step1
medi3_data = medi2.dropna(axis=0)
medi3_data.isnull().sum()

medi3_test = medi2.loc[medi2['ldl_chole'].isnull()]
medi3_test.isnull().sum()
len(medi3_test)


## step2: seperate training/validation data 
from sklearn.model_selection import train_test_split
train_set, valid_set = train_test_split(medi3_data, test_size = 0.3)

train_set.shape
valid_set.shape

sns.pairplot(train_set, x_vars=['TV','Radio','Newspaper'], y_vars='Sales', size=7, aspect=0.7)


import statsmodels.formula.api as sm 
from statsmodels.sandbox.regression.predstd import wls_prediction_std

lm = sm.ols(formula='ldl_chole ~ bp_high+bp_low+total_chole+hdl_chole+BMI', data=train_set).fit()
lm.summary() ## R-squared: 0.572

lm1 = sm.ols(formula='ldl_chole ~ bp_high+total_chole+hdl_chole+blds+hmg+BMI+blds+triglyceride+smoke+drink_yn', data=train_set).fit()
lm1.summary()  ## R-squared: 0.643

lm2 = sm.ols(formula='ldl_chole ~ sex + height+ weight + blds + total_chole + triglyceride + hdl_chole + hmg + creatinine+sgot_ast+sgpt_alt+gamma_gtp+drink_yn+BMI', data=train_set).fit()
lm2.summary()  ## R-squared: 0.643

lm3 = sm.ols(formula='ldl_chole ~ height+ weight + total_chole + hdl_chole+ triglyceride + hmg +BMI', data=train_set).fit()
lm3.summary()  ## R-squared: 0.643



## lm1 prediction & validation 
pred1 = lm1.predict(valid_set)
pred1.head()

actual = valid_set['ldl_chole']
predict1 = pred1
mse_m1 = sum((actual - predict1)**2)/len(actual)
mse_m1
rmse_m1 = (mse_m1)**0.5
rmse_m1


## lm2 prediction & validation
pred2 = lm2.predict(valid_set)
pred2.head()

actual = valid_set['ldl_chole']
predict2 = pred2
mse_m2 = sum((actual - predict2)**2)/len(actual)
mse_m2
rmse_m2 = (mse_m2)**0.5
rmse_m2

## lm3 prediction & validation
pred3 = lm3.predict(valid_set)

actual = valid_set['ldl_chole']
predict3 = pred3
mse_m3 = sum((actual - predict3)**2)/len(actual)
mse_m3
rmse_m3 = (mse_m3)**0.5
rmse_m3


## we choose lm2 model > apply this model for test_data

pred_test = lm2.predict(medi3_test)
pred_test.head()
medi3_test['ldl_chole'] = pred_test
medi3_test

 




################### olig_prote_cd > categorical variable : logistic 
olig_lm = sm.ols('olig_prote_cd ~ bp_high+bp_low+total_chole+hdl_chole+BMI', data=train_set).fit()
olig_lm.summary()


c = ['member_id', 'sex', 'age_group', 'city_code', 'height', 'weight', 'waist', 
     'sight_left', 'sight_right', 'hear_left', 'hear_right', 
     'bp_high', 'bp_low', 'blds', 'total_chole', 'triglyceride',
     'hdl_chole', 'ldl_chole', 'hmg', 'olig_prote_cd', 'creatinine', 
     'sgot_ast', 'sgpt_alt', 'gamma_gtp', 'smoke', 'drink_yn', 'dent_exam_yn','BMI', 'Obesity']

sns.pairplot(train_set, x_vars=['blds','triglyceride'], y_vars='olig_prote_cd', size=7, aspect=0.7)


lm1 = sm.ols(formula='ldl_chole ~ bp_high+total_chole+hdl_chole+blds+hmg+BMI+blds+triglyceride+smoke+drink_yn', data=train_set).fit()
lm1.summary()  ## R-squared: 0.654

lm2 = sm.ols(formula='ldl_chole ~ sex + height+ weight + blds + total_chole + triglyceride + hdl_chole + hmg + creatinine+sgot_ast+sgpt_alt+gamma_gtp+drink_yn+BMI', data=train_set).fit()
lm2.summary()  ## R-squared: 0.654

lm3 = sm.ols(formula='ldl_chole ~ height+ weight + total_chole + hdl_chole+ triglyceride + hmg +BMI', data=train_set).fit()
lm3.summary() 


lm2.predict



'''
missing values:
sight_left         162 > drop
sight_right        160 > drop
hear_left          137 > drop
hear_right         134 > drop

ldl_chole         4242 > regression analysis

olig_pr           4339 >  regression analysis

smoke              382 > fill with the most frequent value
drink_yn           984 > file with the most frequent value

crs_yn          889952 > drop
ttr_yn          617278 > drop 

'''





