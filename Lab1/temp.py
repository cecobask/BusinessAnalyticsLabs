import pandas
import numpy

nesarc_data = pandas.read_csv('nesarc_pds.csv', low_memory=False)
nesarc_data.columns = map(str.upper, nesarc_data.columns)
pandas.set_option('display.float_format',lambda x:'%f'%x)

#number of observations (rows)
print (len(nesarc_data))
# number of variables (columns)
print (len(nesarc_data.columns))

#set PANDAS to show all columns in Data frame
pandas.set_option('display.max_columns', None)

# NICOTINE DEPENDENCE IN THE LAST 12 MONTHS
c1 = nesarc_data["TAB12MDX"].value_counts(sort=False)
print('counts for TAB12MDX - NICOTINE DEPENDENCE IN THE LAST 12 MONTHS, yes=1')
print (c1)
p1 = nesarc_data["TAB12MDX"].value_counts(sort=False, normalize=True)
print('percentages for TAB12MDX - NICOTINE DEPENDENCE IN THE LAST 12 MONTHS, yes=1')
print (p1)

ct1 = nesarc_data.groupby('TAB12MDX').size()
print(ct1)
pt1 = nesarc_data.groupby('TAB12MDX').size() * 100 / len(nesarc_data)
print(pt1)

# CIGARETTE SMOKING STATUS
c2 = nesarc_data["CHECK321"].value_counts(sort=False)
print('counts for CHECK321 - CIGARETTE SMOKING STATUS, yes=1')
print (c2)
p2 = nesarc_data["CHECK321"].value_counts(sort=False, normalize=True)
print('percentages for CHECK321 - CIGARETTE SMOKING STATUS, yes=1')
print (p2)

# USUAL FREQUENCY WHEN SMOKED CIGARETTES
c3 = nesarc_data["S3AQ3B1"].value_counts(sort=False)
print('counts for S3AQ3B1 - CIGARETTE SMOKING STATUS')
print (c3)
p3 = nesarc_data["S3AQ3B1"].value_counts(sort=False, normalize=True)
print('percentages for S3AQ3B1 - CIGARETTE SMOKING STATUS')
print (p3)

# USUAL QUANTITY WHEN SMOKED CIGARETTES
pandas.to_numeric(nesarc_data['S3AQ3C1'],errors='coerce')
c4 = nesarc_data["S3AQ3C1"].value_counts(sort=True)
print('counts for S3AQ3C1 - USUAL QUANTITY WHEN SMOKED CIGARETTES')
print (c4)
p4 = nesarc_data["S3AQ3C1"].value_counts(sort=True, normalize=True)
print('percentages for S3AQ3C1 - USUAL QUANTITY WHEN SMOKED CIGARETTES')
print (p4)