import pandas as pd
import numpy

# Load dataset.
addhealth_data = pd.read_csv('../addhealth_pds.csv', low_memory=False)

# Restrict dataset to observations that know their parents.
dataset = addhealth_data[(addhealth_data['H1NF1'] == 7) & (addhealth_data['H1NM1'] == 7)].copy()
# Set display results to 0 decimal points.
pd.set_option("display.precision", 0)

print('\nNumber of people that know their mother:')
print(dataset['H1NM1'].value_counts().sum())
print('\nNumber of people that know their father:')
print(dataset['H1NF1'].value_counts().sum())

print("\nHow far in school did the mother go?")
# 1 eighth grade or less
# 2 more than eighth grade, but did not graduate from high school
# 3 went to a business, trade, or vocational school instead of high school
# 4 high school graduate
# 5 completed a GED
# 6 went to a business, trade, or vocational school after high school
# 7 went to college, but did not graduate
# 8 graduated from a college or university
# 9 professional training beyond a four-year college or university
# 10 She never went to school.
dataset['H1RM1'] = dataset['H1RM1'].replace([11, 12, 96, 97, 98], numpy.nan)
print(dataset['H1RM1'].value_counts())

print("\nHow far in school did the father go?")
# Answer codes are identical to previous printout.
dataset['H1RF1'] = dataset['H1RF1'].replace([11, 12, 96, 97, 98], numpy.nan)
print(dataset['H1RF1'].value_counts())
