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


# medi = pd.read_csv('C:\\Users\\Hyesu KIM\\Documents\\Python_Hyesu\\Python_project\\Medical\\medical_2014.csv', engine='python')
medi = pd.read_csv('C:\\Users\\202-22\\Documents\\Python - Hyesu\\Project\\Medical\\medical_2014.csv', engine='python')
medi.head()


# 02.Data manupulation

medi.shape  ## 10,000,000 rows x 31 columns 

medi.index
medi.columns

'''
Original columns names: 
'기준년도', '가입자일련번호', '성별코드', '연령대코드(5세단위)', '시도코드', '신장(5Cm단위)',
'체중(5Kg 단위)', '허리둘레', '시력(좌)', '시력(우)', '청력(좌)', '청력(우)', '수축기혈압',
'이완기혈압', '식전혈당(공복혈당)', '총콜레스테롤', '트리글리세라이드', 'HDL콜레스테롤', 'LDL콜레스테롤',
'혈색소', '요단백', '혈청크레아티닌', '(혈청지오티)AST', '(혈청지오티)ALT', '감마지티피', '흡연상태',
'음주여부', '구강검진 수검여부', '치아우식증유무', '치석유무', '데이터 기준일자'
'''
       
       
medi.rename(index=str, columns={'기준년도': 'year', '가입자일련번호': 'member_id', '성별코드': 'sex', 
                               '연령대코드(5세단위)': 'age group', '시도코드': 'city_code', 
                               '신장(5Cm단위)': 'height', '체중(5Kg 단위)': 'weight', '허리둘레':'waist',
                               '시력(좌)': 'sight_left', '시력(우)': 'sight_right', '청력(좌)': 'hear_left', '청력(우)':'hear_right',
                               '수축기혈압':'bp_high', '이완기혈압': 'bp_low', '식전혈당(공복혈당)': 'blds',
                               '총콜레스테롤':'total_chole', '트리글리세라이드':'triglyceride', 'HDL콜레스테롤':'hdl_chole',
                               'LDL콜레스테롤':'ldl_chole', '혈색소':'hmg', '요단백':'olig_pr', 
                               '혈청크레아티닌':'creatinine', '(혈청지오티)AST':'sgot_ast', '(혈청지오티)ALT':'sgpt_alt',
                               '감마지티피':'gamma_gtp', '흡연상태':'smoke', '음주여부':'drink_yn', 
                               '구강검진 수검여부':'dent_exam_yn', '치아우식증유무':'crs_yn', '치석유무':'ttr_yn', '데이터 기준일자':'data_date'})

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
medi1.dtypes
medi1.describe()
"""
- sex (male:1, female:2)
- age_group (total: 14 groups, only from 5-18 group)  # need to see the data visualisation to know why the max is 18
- height 
- weight (min: 30kg ??? )  ## to check
- waist (max: 129cm ????)  ## to check
- sight_left (0.1~2.5, sight under 0.1 = 0.1, blindness:9.9)
- sight_right 



"""



## continuous variables to categorial variables
# df['col_name'] = df['col_name'].astype(object)
medi1['sex'] = medi1['sex'].astype('category')
medi1['sex'].describe()

medi1['age_group'] = medi1['age_group'].astype('category')
medi1['city_code'] = medi1['city_code'].astype('category')
medi1['olig_pr'] = medi1['olig_pr'].astype('category')
medi1['smoke'] = medi1['smoke'].astype('category')
medi1['drink_yn'] = medi1['drink_yn'].astype('category')
medi1['dent_exam_yn'] = medi1['dent_exam_yn'].astype('category')
medi1['crs_yn'] = medi1['crs_yn'].astype('category')
medi1['ttr_yn'] = medi1['ttr_yn'].astype('category')

medi1['age_group'].describe()   ## 14 categories out of 18, top category is 9(60~64 years old)
medi1['city_code'].describe()   ## 17 city codes out of 17, top category is 41(GeongGi-do)
medi1['smoke'].describe()       ## top category is 1 (non-smoking)
medi1['drink_yn'].describe()    ## yes > no
medi1['dent_exam_yn'].describe() ## yes < no
medi1['crs_yn'].describe()    ## When summing data, NA (missing) values will be treated as zero
medi1['ttr_yn'].describe()    ## When summing data, NA (missing) values will be treated as zero




## find out if there is missing value and where
medi1.info()   ## count non-null cells in each column

medi1.isnull().any()  ## to find out if there is NaNs 
medi1.isnull().sum()  ## to count how many NaNs 

drink_nan = medi1[medi1['drink_yn'].isnull()]
print(drink_nan)

smoke_nan = medi1[medi1['smoke'].isnull()]
olig_nan = medi1[medi1['olig_pr'].isnull()]
ldlc_nan = medi1[medi1['ldl_chole'].isnull()]




#03. histogram
import matplotlib.pyplot as plt

## histogram of continuus variables
medi1['total_chole'].plot(kind='hist', bins=100)
plt.xlim(xmin = 0, xmax = 500)
plt.xlabel('Total_Cholestrol')

medi1['bp_high'].plot(kind='hist', bins=100)
plt.xlim(xmin = 0, xmax = 200)
plt.xlabel('Highest Bloodpressure')

medi1['bp_low'].plot(kind='hist', bins=100)
plt.xlim(xmin = 0, xmax = 150)
plt.xlabel('Lowest Bloodpressure')

## boxplot of continuus variables
plt.boxplot(medi1['total_chole'])
plt.boxplot(medi1['bp_high'])
plt.boxplot(medi1['bp_low'])




## histogram of categorial variables
plt.bar(medi1['sex'], color=("blue", "red"))  ## not working... why ???





