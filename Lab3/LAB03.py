import pandas
import numpy

# Load dataset.
nesarc_data = pandas.read_csv('../nesarc_pds.csv', low_memory=False)

# Convert strings to numbers.
nesarc_data['TAB12MDX'] = pandas.to_numeric(nesarc_data['TAB12MDX'], errors='ignore')
nesarc_data['CHECK321'] = pandas.to_numeric(nesarc_data['CHECK321'], errors='ignore')
nesarc_data['S3AQ3B1'] = pandas.to_numeric(nesarc_data['S3AQ3B1'], errors='ignore')
nesarc_data['S3AQ3C1'] = pandas.to_numeric(nesarc_data['S3AQ3C1'], errors='ignore')

# Restrict to those observations that are between 18 and 25 and smoke now.
subset1 = nesarc_data[(nesarc_data['AGE'] >= 18) & (nesarc_data['AGE'] <= 25) & (nesarc_data['CHECK321'] == '1')]
subset2 = subset1.copy()

# Convert field to number.
subset2['S3AQ3B1'] = pandas.to_numeric(subset2['S3AQ3B1'])
subset2['S3AQ3C1'] = pandas.to_numeric(subset2['S3AQ3C1'])

# Set undefined and unknown values to nan.
subset2['S3AQ3B1'] = subset2['S3AQ3B1'].replace(9, numpy.nan)
subset2['S3AQ3C1'] = subset2['S3AQ3C1'].replace(99, numpy.nan)
subset2['S2AQ3'] = subset2['S2AQ3'].replace(9, numpy.nan)
subset2['S2AQ8A'] = subset2['S2AQ8A'].replace(' ', numpy.NaN)

subset2.loc[(subset2['S2AQ3'] != 9) & (subset2['S2AQ8A'].isnull()), 'S2AQ8A'] = 11
recode1 = {1: 6, 2: 5, 3: 4, 4: 3, 5: 2, 6: 1}
subset2['USFREQ'] = subset2['S3AQ3B1'].map(recode1)
recode2 = {1: 30, 2: 22, 3: 14, 4: 5, 5: 2.5, 6: 1}
subset2['USFREQMO'] = subset2['S3AQ3B1'].map(recode2)

# Number of cigarettes per month smoked by respondents.
subset2['S3AQ3C1'] = pandas.to_numeric(subset2['S3AQ3C1'])
subset2['NUMCIGMO_EST'] = subset2['USFREQMO'] * subset2['S3AQ3C1']
print(subset2['NUMCIGMO_EST'].value_counts(sort=True, dropna=False))

subset3 = subset2[['IDNUM', 'S3AQ3C1', 'USFREQMO', 'NUMCIGMO_EST']]
print(subset3.head(25))

# Test if data is correct by calculating a known value.
test = subset3['USFREQMO'] * subset3['S3AQ3C1']
print(test.value_counts(sort=False, dropna=False))

# Split column into quartiles.
print('\nAGE - 4 Categories - quartiles')
subset2['AGEGROUP'] = pandas.qcut(subset2.AGE, 4, labels=['1=25%tile', '2=50%tile', '3=75%tile', '4=100%tile'])
c14 = subset2['AGEGROUP'].value_counts(sort=False, dropna=True)
print(c14)

# Categorise variable based on customised splits using the cut() functions.
# Splits into three groups: 18-20, 21-22, and 23-25
subset2['AGEGROUP2'] = pandas.cut(subset2.AGE, [17, 20, 22, 25], labels=['18-20', '21-22', '23-25'])
c15 = subset2['AGEGROUP2'].value_counts(sort=False, dropna=True)
print(c15)

# Compare the different groups of ages using a crosstab.
print(pandas.crosstab(subset2['AGEGROUP2'], subset2['AGE']))

# Print AGEGROUP2 values as percentages, rounded to 2 decimal places.
c16 = subset2['AGEGROUP2'].value_counts(normalize=True).round(decimals=2)
print(c16)
