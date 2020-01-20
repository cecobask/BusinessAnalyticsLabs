import pandas
import numpy

# Load dataset.
nesarc_data = pandas.read_csv('../nesarc_pds.csv', low_memory=False)

# Convert strings to numbers.
nesarc_data['TAB12MDX'] = pandas.to_numeric(nesarc_data['TAB12MDX'],errors='ignore')
nesarc_data['CHECK321'] = pandas.to_numeric(nesarc_data['CHECK321'],errors='ignore')
nesarc_data['S3AQ3B1'] = pandas.to_numeric(nesarc_data['S3AQ3B1'], errors='ignore')
nesarc_data['S3AQ3C1'] = pandas.to_numeric(nesarc_data['S3AQ3C1'], errors='ignore')

# Restrict to those observations that are between 18 and 25 and smoke now.
subset1 = nesarc_data[(nesarc_data['AGE']>=18) & (nesarc_data['AGE']<=25) & (nesarc_data['CHECK321']=='1')]

print('counts for AGE')
c1 = subset1['AGE'].value_counts(sort=True)
print(c1)

print('percentages for AGE')
c2 = subset1['AGE'].value_counts(sort=True, normalize=True)
print(c2)

subset2 = subset1.copy()

print('Number of observations in subset 2')
print(len(subset2))

# Convert field to number.
subset2['S3AQ3B1'] = pandas.to_numeric(subset2['S3AQ3B1'])
subset2['S3AQ3C1'] = pandas.to_numeric(subset2['S3AQ3C1'])

# Counts for S3AQ3B1.
print('counts for S3AQ3B1 - usual frequency when smoked cigarettes')
c7 = subset2["S3AQ3B1"].value_counts(sort=True)
print (c7)

# Set '9' values to nan.
subset2['S3AQ3B1'] = subset2['S3AQ3B1'].replace(9,numpy.nan)
subset2['S3AQ3C1'] = subset2['S3AQ3C1'].replace(99,numpy.nan)
subset2['S2AQ3'] = subset2['S2AQ3'].replace(9,numpy.nan)
subset2['S2AQ8A'] = subset2['S2AQ8A'].replace(' ', numpy.NaN)

print((subset2['S3AQ3B1']==9).sum())
print(subset2['S3AQ3B1'].isnull().sum())
print('counts for S3AQ3B1 - usual frequency when smoked cigarettes')

c8 = subset2['S3AQ3B1'].value_counts(sort=True, dropna=False)
print (c8)

c9 = subset2['S3AQ3C1'].value_counts(sort=True, dropna=False)
print(c9)

c10 = subset2['S2AQ3'].value_counts(sort=True, dropna=False)
print(c10)

c11 = subset2['S2AQ8A'].value_counts(sort=True, dropna=False)
print(c11)

# First see if there are any nulls
print((subset2['S2AQ8A'].isnull()).sum())
# Next see if there are any empty values
print((subset2['S2AQ8A']=="").sum())
# Next see if there are any that contain a space
print((subset2['S2AQ8A']==" ").sum())

subset2.loc[(subset2['S2AQ3']!=9) & (subset2['S2AQ8A'].isnull()),'S2AQ8A']=11
c12 = subset2['S2AQ8A'].value_counts(sort=True, dropna=False)
print(c12)

c13 = subset2['S3AQ3B1'].value_counts(sort=True, dropna=False)
print(c13)

recode1 = {1: 6, 2: 5, 3: 4, 4: 3, 5: 2, 6: 1}
subset2['USFREQ'] = subset2['S3AQ3B1'].map(recode1)
c14 = subset2['USFREQ'].value_counts(sort=True, dropna=False)
print(c14)

recode2 = {1: 30, 2: 22, 3: 14, 4: 5, 5: 2.5, 6: 1}
subset2['USFREQMO'] = subset2['S3AQ3B1'].map(recode2)
c15 = subset2['USFREQMO'].value_counts(sort=True, dropna=False)
print(c15)