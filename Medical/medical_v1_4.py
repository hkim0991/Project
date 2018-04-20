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
                               'LDL콜레스테롤':'ldl_chole', '혈색소':'hmg', '요단백':'olig_pr', 
                               '혈청크레아티닌':'creatinine', '(혈청지오티)AST':'sgot_ast', '(혈청지오티)ALT':'sgpt_alt',
                               '감마지티피':'gamma_gtp', '흡연상태':'smoke', '음주여부':'drink_yn', 
                               '구강검진 수검여부':'dent_exam_yn', '치아우식증유무':'crs_yn', '치석유무':'ttr_yn', '데이터 기준일자':'data_date'})

medi1.head(n=10)
medi1.dtypes # to check data type
medi1.describe() # to summarize the data
"""
- sex (male:1, female:2)
- age_group (total: 14 groups, only from 5-18 group)  # need to see the data visualisation to know why the max is 18
- height 
- weight (30~140, min: 30kg ??? )  ## to check
- waist (51~129, max: 129cm ????)  ## to check
- sight_left (0.1~2.5, sight under 0.1 = 0.1, blindness:9.9)
- sight_right (0.1~2.5, sight under 0.1 = 0.1, blindness:9.9)
- creatinine (normal: 0.8~1.7, 0.1~98, max=98 ????) ## to check
- sgot_ast (normal: 0~40 IU/L, 1~999, max=999 ???)  ## to check
- sgpt_alt (normal: 0~40 IU/L, 1~999, max=999 ???)  ## to check 
- gamma_gtp (normal: male - 11~63IU/L, female- 8~35IU/Lm, 1~999, max= 999 ???)
- smoke (no-smoking:1, quitted:2, smoke:3)
- drink_yn (no:0, yes:1)
- dental examination (no:0, yes:1)
- crs_yn (no:0, yes:1)
- ttr_tn (no:0, yes:1)
"""



## continuous variables to categorial variables
# df['col_name'] = df['col_name'].astype(object)
def to_category(variable_name):
    list = ['sex', 'age_group', 'city_code', 'olig_pr', 'smoke', 'drink_yn', 'dent_exam_yn', 'crs_yn','ttr_yn']
    if variable_name in list:
        medi1[variable_name] = medi1[variable_name].astype('category')


'''
medi1['sex'] = medi1['sex'].astype('category')
medi1['age_group'] = medi1['age_group'].astype('category')
medi1['city_code'] = medi1['city_code'].astype('category')
medi1['olig_pr'] = medi1['olig_pr'].astype('category')
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

medi1.info()


## find out if there is missing value and where
medi1.info()   ## count non-null cells in each column

medi1.isnull().any()  ## to find out if there is NaNs (True or False) 
medi1.isnull().sum()  ## to count how many NaNs   

## Question is how to handlie the missing values in categorical variables?

drink_nan = medi1[medi1['drink_yn'].isnull()]  # to show where is the missing value
print(drink_nan)

smoke_nan = medi1[medi1['smoke'].isnull()]
olig_nan = medi1[medi1['olig_pr'].isnull()]
ldlc_nan = medi1[medi1['ldl_chole'].isnull()]


## to count the numbers in each category value
age_freq = medi1['age_group'].value_counts()
age_freq


'''
freq = tdf['Embarked'].value_counts() 
most_freq =  tdf['Embarked'].mode()
tdf2['Embarked'].fillna(most_freq[0], inplace=True)
'''



#03. data exploration:plot, 
## histogram of continuus variables
medi1['total_chole'].plot(kind='hist', bins=100) ## normal: 150~250
plt.xlabel('Total_Cholestrol')

medi1['bp_high'].plot(kind='hist', bins=100)
plt.xlim(xmin = 50, xmax = 200)
plt.xlabel('Highest Bloodpressure')

medi1['bp_low'].plot(kind='hist', bins=100)
plt.xlim(xmin = 30, xmax = 130)
plt.xlabel('Lowest Bloodpressure')

## boxplot of continuus variables

plt.boxplot(medi1['total_chole'])
plt.boxplot(medi1['bp_high'])
plt.boxplot(medi1['bp_low'])


sns.boxplot(medi1['weight'])
sns.boxplot(medi1['height'])
sns.boxplot(medi1['waist'])
sns.boxplot(medi1['total_chole'])
 
sns.boxplot(medi1['ldl_chole'])
sns.boxplot(medi1['olig_pr'])

sns.boxplot(medi1['creatinine'])
sns.boxplot(medi1['sgot_ast'])
sns.boxplot(medi1['sgpt_alt'])
sns.boxplot(medi1['gamma_gtp'])

sns.


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


sns.countplot(medi1['olig_pr'])
plt.xlabel('olig_pr', fontsize=10)
plt.ylabel('Numbers', fontsize=10)
plt.title('olig_pr in 2014')


sns.countplot(medi1['drink_yn'])
plt.xlabel('Drinking status', fontsize=10)
plt.ylabel('Numbers', fontsize=10)
plt.title('Drinking Status in 2014')



'''
NOT WORKING
plt.bar(medi1['smoke'])
plt.xlabel('Smoking status', fontsize=5)
plt.ylabel('Numbers', fontsize=5)
plt.title('Smoking Status in 2014')
plt.show()


## histogram of categorial variables
plt.bar(medi1['sex'], color=("blue", "red"))  ## not working... why ???

'''''



