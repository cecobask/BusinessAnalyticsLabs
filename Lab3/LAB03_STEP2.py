import pandas
import numpy

# Load dataset.
pandas.set_option('display.max_columns', 20)
data = pandas.read_csv('../AddHealth/addhealth_pds.csv', low_memory=False)

# Extract columns relating to ethnicity.
data['H1GI4'] = pandas.to_numeric(data['H1GI4'])
data['H1GI6A'] = pandas.to_numeric(data['H1GI6A'])
data['H1GI6B'] = pandas.to_numeric(data['H1GI6B'])
data['H1GI6C'] = pandas.to_numeric(data['H1GI6C'])
data['H1GI6D'] = pandas.to_numeric(data['H1GI6D'])

# Deal with missing data.
data['H1GI4'] = data['H1GI4'].replace([6, 8], numpy.nan)
data['H1GI6A'] = data['H1GI6A'].replace([6, 8], numpy.nan)
data['H1GI6B'] = data['H1GI6B'].replace([6, 8], numpy.nan)
data['H1GI6C'] = data['H1GI6C'].replace([6, 8], numpy.nan)
data['H1GI6D'] = data['H1GI6D'].replace([6, 8], numpy.nan)

# Create a new variable that indicates number of race or ethnicity variables.
data['NUMETHNIC'] = data['H1GI4'] + data['H1GI6A'] + data['H1GI6B'] + data['H1GI6C'] + data['H1GI6D']
print('Counts for new variable.')
print(data['NUMETHNIC'].value_counts(sort=True, dropna=True).head(5))


def ethnicity(row):
    if row['NUMETHNIC'] > 1:
        return 1
    if row['H1GI4'] == 1:
        return 2
    if row['H1GI6A'] == 1:
        return 3
    if row['H1GI6B'] == 1:
        return 4
    if row['H1GI6C'] == 1:
        return 5
    if row['H1GI6D'] == 1:
        return 6


# Create new column.
data['ETHNICITY'] = data.apply(lambda row: ethnicity(row), axis=1)
print(data['ETHNICITY'].value_counts(sort=False, dropna=True))

# Create a subset with selected columns.
subset2 = data[['AID', 'H1GI4', 'H1GI6A', 'H1GI6B', 'H1GI6C', 'H1GI6D', 'NUMETHNIC', 'ETHNICITY']]
print(subset2.head(25))
