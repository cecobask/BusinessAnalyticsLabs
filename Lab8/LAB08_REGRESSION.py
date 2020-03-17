# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 09:20:52 2019

@author: Brenda_MJtech
"""

import pandas
import numpy
import statsmodels.api as sm
import statsmodels.formula.api as smf

# load dataset from the csv file in the dataframe called nesarc_data
nesarc_data = pandas.read_csv('../nesarc_pds.csv', low_memory=False)

# set PANDAS to show all columns in Data frame
pandas.set_option('display.max_columns', None)

# set PANDAS to show all rows in Data frame
pandas.set_option('display.max_rows', None)

# converting strings to numeric data for better output

nesarc_data['TAB12MDX'] = pandas.to_numeric(nesarc_data['TAB12MDX'], errors='ignore')
nesarc_data['CHECK321'] = pandas.to_numeric(nesarc_data['CHECK321'], errors='ignore')
nesarc_data['S3AQ3B1'] = pandas.to_numeric(nesarc_data['S3AQ3B1'], errors='ignore')
nesarc_data['S3AQ3C1'] = pandas.to_numeric(nesarc_data['S3AQ3C1'], errors='ignore')
nesarc_data['AGE'] = pandas.to_numeric(nesarc_data['AGE'], errors='ignore')

# restrict to those observations that are between 18 and 25 and smoke now
subset1 = nesarc_data[(nesarc_data['AGE'] >= 18) & (nesarc_data['AGE'] <= 25) & (nesarc_data['CHECK321'] == '1')]

subset2 = subset1.copy()

# replacing missing data
# counts for S3AQ3B1
print('counts for S3AQ3B1 - usual frequency when smoked cigarettes')
c7 = subset1['S3AQ3B1'].value_counts(sort=True)
print(c7)

# replace blanks with Nan
subset2['TAB12MDX'] = subset2['TAB12MDX'].replace(" ", numpy.NaN)
subset2['S3AQ3B1'] = subset2['S3AQ3B1'].replace(" ", numpy.NaN)
subset2['S3AQ3C1'] = subset2['S3AQ3C1'].replace(" ", numpy.NaN)

# ensure the variables are number data type first
subset2['S3AQ3B1'] = pandas.to_numeric(subset2['S3AQ3B1'])
subset2['S3AQ3C1'] = pandas.to_numeric(subset2['S3AQ3C1'])
subset2['TAB12MDX'] = pandas.to_numeric(subset2['TAB12MDX'])

# replace the value 9 in S3AQ3B1 with Nan to signify missing data
subset2['S3AQ3B1'] = subset2['S3AQ3B1'].replace(9, numpy.NaN)

print((subset2['S3AQ3B1'] == 9).sum())
print(subset2['S3AQ3B1'].isnull().sum())

# counts for S3AQ3B1 after set to Nan
print(subset2.describe())
print('counts for S3AQ3B1 - usual frequency when smoked cigarettes')
c8 = subset2['S3AQ3B1'].value_counts(sort=True, dropna=False)
print(c8)

# recoding values

# first create the dictionary to recode
recode1 = {1: 6, 2: 5, 3: 4, 4: 3, 5: 2, 6: 1}

# next use the map funciton to replace values using the recode dictionary
subset2['USFREQ'] = subset2['S3AQ3B1'].map(recode1)

# Now recode to a quantitative value based on an estimate of how many times per month each person smokes
recode2 = {1: 30, 2: 22, 3: 14, 4: 5, 5: 2.5, 6: 1}

subset2['USFREQMO'] = subset2['S3AQ3B1'].map(recode2)

# create new secondary variable to hold number of cigarettes per month
subset2['NUMCIGMO_EST'] = subset2['USFREQMO'] * subset2['S3AQ3C1']
subset2['NUMCIGMO_EST'] = subset2['NUMCIGMO_EST'].replace(" ", numpy.NaN)
subset2['NUMCIGMO_EST'] = pandas.to_numeric(subset2['NUMCIGMO_EST'])

# Multivariate linear regression
print('OLS regression model for the association between nicotine dependence, major depression and number of cigarettes smoked per month')
reg2 = smf.ols('NUMCIGMO_EST ~ TAB12MDX + MAJORDEPLIFE', data=subset2).fit()
print(reg2.summary())

# Subset of only two variables.
subset2 = subset2[['TAB12MDX', 'NUMCIGMO_EST']].dropna()

# Sample size.
print(len(subset2))

# regression for association between nicotine dependence and Number of cigarettes smoked per month
print('OLS regression model for the association between nicotine dependence and number of cigarettes smoked per month')
reg1 = smf.ols('NUMCIGMO_EST ~ TAB12MDX', data=subset2).fit()
print(reg1.summary())

print("Mean")
ds1 = subset2.groupby('TAB12MDX').mean()
print(ds1)

print("STD")
ds2 = subset2.groupby('TAB12MDX').std()
print(ds2)
