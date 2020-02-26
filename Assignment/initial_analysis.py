import pandas
import numpy

# Load dataset.
addhealth_data = pandas.read_csv('../addhealth_pds.csv', low_memory=False)

# Restrict dataset to observations who know their parents.
dataset = addhealth_data[addhealth_data['H1NF1'].isin([1, 7]) & addhealth_data['H1NM1'].isin([1, 7])]
print('Number of people that know their father:')
print(dataset['H1NF1'].value_counts().sum())
print('Number of people that know their mother:')
print(dataset['H1NM1'].value_counts().sum())
