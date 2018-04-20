# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 13:33:59 2018

@author: 202-22
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

medi = pd.read_csv('C:\\Users\\202-22\\Documents\\Python - Hyesu\\Project\\Medical\\medical_2014.csv', engine='python')

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

list = ['sex', 'age_group', 'city_code', 'olig_pr', 'smoke', 'drink_yn', 'dent_exam_yn', 'crs_yn','ttr_yn']
for i in list:
    medi1[i] = medi1[i].astype('category')
    print("the type of", i, medi1[i].dtypes)
    
    

## countinous variables
# 1. Total Cholestrol
# 1.1 boxplot
plt.boxplot(medi1['total_chole'])

# 1.2 Histogram
medi1['total_chole'].plot(kind='hist', bins=100) ## normal: 150~250
plt.xlabel('Total_Cholestrol')

# count how many outliner
out_total_chole = medi1['total_chole'].value_counts()    ## 250 이상일 경우에만 count
out_total_chole 