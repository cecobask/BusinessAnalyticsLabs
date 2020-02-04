import pandas
import numpy
import seaborn
import matplotlib.pyplot as plt

# set PANDAS to show all columns and rows in Data frame.
pandas.set_option('display.max_columns', None)
pandas.set_option('display.max_rows', None)

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

# Convert fields to number.
subset2['S3AQ3B1'] = pandas.to_numeric(subset2['S3AQ3B1'])
subset2['S3AQ3C1'] = pandas.to_numeric(subset2['S3AQ3C1'])

# Set undefined and unknown values to nan.
subset2['S3AQ3B1'] = subset2['S3AQ3B1'].replace(9, numpy.nan)
subset2['S3AQ3C1'] = subset2['S3AQ3C1'].replace(99, numpy.nan)
subset2['S2AQ3'] = subset2['S2AQ3'].replace(9, numpy.nan)
subset2['S2AQ8A'] = subset2['S2AQ8A'].replace(' ', numpy.NaN)

recode1 = {1: 30, 2: 22, 3: 14, 4: 5, 5: 2.5, 6: 1}
subset2['USFREQMO'] = subset2['S3AQ3B1'].map(recode1)

# Number of cigarettes per month smoked by respondents.
subset2['NUMCIGMO_EST'] = subset2['USFREQMO'] * subset2['S3AQ3C1']

print("Value counts for NUMCIGMO_EST")
print(subset2['NUMCIGMO_EST'].value_counts(sort=True, dropna=False))
print("Percentages for NUMCIGMO_EST")
print(subset2['NUMCIGMO_EST'].value_counts(sort=True, dropna=False, normalize=True))

print("Value counts for TAB12MDX")
print(subset2['TAB12MDX'].value_counts(sort=True, dropna=False))
print("Percentages for TAB12MDX")
print(subset2['TAB12MDX'].value_counts(sort=True, dropna=False, normalize=True))

print("Describe number of cigarettes per month smoked by respondents")
print(subset2["NUMCIGMO_EST"].describe())

# Convert variable.
subset2['TAB12MDX'] = subset2['TAB12MDX'].astype('category')

# Display charts using matplotlib and seaborn.
seaborn.countplot(x='TAB12MDX', data=subset2)
plt.xlabel('Nicotine Dependence past 12 months')
plt.title('Nicotine Dependence in the past 12 months among young adult smokers in the Nesarc study', wrap=True)
plt.show()

seaborn.distplot(subset2['NUMCIGMO_EST'].dropna(), kde=False)
plt.xlabel('Number of cigarettes per month')
plt.title('Estimated number of cigarettes per month among young adult smokers in the Nesarc study',
          wrap=True)
plt.show()

# Create bins for var NUMCIGMO_EST.
subset2["NUMCIGMO_EST_BINS"] = pandas.cut(subset2['NUMCIGMO_EST'],
                                          [0, 200, 400, 600, 800, 1000, 4000],
                                          labels=['1-200', '200-400', '400-600', '600-800', '800-1000', '1000-4000'])
print(subset2["NUMCIGMO_EST_BINS"].value_counts(sort=False, dropna=True))

# Display a chart of new variable NUMCIGMO_EST_BINS.
plt.figure(figsize=(9, 6))  # change size of chart.
seaborn.countplot(x='NUMCIGMO_EST_BINS', data=subset2)
plt.title('Nicotine dependence in the past 12 months among young adult smokers in the Nesarc study',
          wrap=True)
plt.xlabel('Nicotine dependence past 12 months')
plt.show()

# Print descriptive statistics about TAB12MDX.
print(pandas.to_numeric(subset2['TAB12MDX']).describe())

# Variable that stores number of cigarette packs smoked per month.
subset2['PACKSPERMONTH'] = subset2['NUMCIGMO_EST'] / 20

# Binned categories for PACKSPERMONTH.
subset2['PACKCATEGORY'] = pandas.cut(subset2.PACKSPERMONTH,
                                     [0, 5, 10, 20, 30, 147])
subset2['PACKCATEGORY'] = subset2['PACKCATEGORY'].astype('category')

print('Describe nicotine dependence')
desc3 = subset2.groupby('PACKCATEGORY').size()
print(desc3)

# Bivariate bar chart.
subset2['TAB12MDX'] = pandas.to_numeric(subset2['TAB12MDX'])
seaborn.catplot(x='PACKCATEGORY', y='TAB12MDX', data=subset2, kind='bar', ci=None)
plt.xlabel('Packs per month')
plt.ylabel('Proportion Nicotine Dependence')
plt.show()


def SMOKEGRP(row):
    # Nicotine dependent.
    if row['TAB12MDX'] == 1:
        return 1
    # Daily smoker.
    elif row['USFREQMO'] == 30:
        return 2
    # All other young adult smokers.
    else:
        return 3


# Create groups of smokers.
subset2['SMOKEGRP'] = subset2.apply(lambda row: SMOKEGRP(row), axis=1)
print(subset2['SMOKEGRP'].value_counts(normalize=True))

# Univariate chart for variable SMOKEGRP.
seaborn.countplot(x='SMOKEGRP', data=subset2)
plt.xlabel('Smoking groups')
plt.show()


def DAILY(row):
    if row['USFREQMO'] == 30:
        return 1
    elif row['USFREQMO'] != 30:
        return 0


# Divide smokers into daily or non-daily.
subset2['DAILY'] = subset2.apply(lambda row: DAILY(row), axis=1)
print(subset2.groupby('DAILY').size())

# Plot the relationship between ETHRACE2A and DAILY variables.
subset2['ETHRACE2A'] = subset2['ETHRACE2A'].astype('category')
subset2['ETHRACE2A'] = subset2['ETHRACE2A'].cat.rename_categories(
    ['White', 'Black', 'NatAm', 'Asian', 'Hispanic'])
seaborn.catplot(x='ETHRACE2A', y='DAILY', data=subset2, kind='bar', ci=None)
plt.xlabel('Ethnic Group')
plt.ylabel('Proportion Daily Smokers')
plt.show()

# Counts for daily and non-daily smokers per ethnic group.
print(subset2.groupby(['DAILY', 'ETHRACE2A']).size())
