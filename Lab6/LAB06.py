import numpy
import pandas
import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi

nesarc_data = pandas.read_csv('../nesarc_pds.csv', low_memory=False)

# Setting missing spaces in text data.
nesarc_data['CHECK321'] = nesarc_data['CHECK321'].replace(' ', numpy.nan)
nesarc_data['S3AQ3B1'] = nesarc_data['S3AQ3B1'].replace(' ', numpy.nan)
nesarc_data['S3AQ3C1'] = nesarc_data['S3AQ3C1'].replace(' ', numpy.nan)

# Setting variables you will be working with to numeric.
nesarc_data['CHECK321'] = pandas.to_numeric(nesarc_data['CHECK321'])
nesarc_data['S3AQ3B1'] = pandas.to_numeric(nesarc_data['S3AQ3B1'])
nesarc_data['S3AQ3C1'] = pandas.to_numeric(nesarc_data['S3AQ3C1'])

# Subset data to young adults age 18 to 26 who have smoked in the past 12 months.
sub1 = nesarc_data.copy()
sub1 = sub1[(sub1['AGE'] >= 18) & (sub1['AGE'] <= 26) & (sub1['CHECK321'] == 1)]

# Setting missing numerical data.
sub1['S3AQ3B1'] = sub1['S3AQ3B1'].replace(9, numpy.nan)
sub1['S3AQ3C1'] = sub1['S3AQ3C1'].replace(99, numpy.nan)

# Recoding number of days smoked in the past month.
recode1 = {1: 30, 2: 22, 3: 14, 4: 5, 5: 2.5, 6: 1}
sub1['USFREQMO'] = sub1['S3AQ3B1'].map(recode1)

# Test that the mapping took place.
print(sub1[['S3AQ3B1', 'USFREQMO']])

# Create a secondary variable multiplying the days smoked/month and the number of cig/per day.
sub1['NUMCIGMO_EST'] = sub1['USFREQMO'] * sub1['S3AQ3C1']

print(sub1['NUMCIGMO_EST'])
print(sub1.groupby('NUMCIGMO_EST').size())

# Using OLS function for calculating the F-statistic and associated p-value.
model1 = smf.ols(formula='NUMCIGMO_EST ~ C(MAJORDEPLIFE)', data=sub1).fit()
print(model1.summary())

# Subset of only two variables.
sub2 = sub1[['NUMCIGMO_EST', 'MAJORDEPLIFE']].dropna()

# Sample size.
print(len(sub2))

# Calculate the means and standard deviations for monthly smoking for each category of MAJORDEPLIFE.
print('Means and Standard Deviations for MAJORDEPLIFE by major depression status:')
print(sub2.groupby('MAJORDEPLIFE').mean())
print(sub2.groupby('MAJORDEPLIFE').std())

# Create a new subset which only contains required variables for analysis.
sub3 = sub1[['NUMCIGMO_EST', 'ETHRACE2A']].dropna()

# Using OLS function for calculating the F-statistic and associated p-value.
model2 = smf.ols(formula='NUMCIGMO_EST ~ C(ETHRACE2A)', data=sub3).fit()
print(model2.summary())

print('Means for ETHRACE2A by ethnic race.')
print(sub3.groupby('ETHRACE2A').mean())

print('Standard Deviations for ETHRACE2A by ethnic race.')
print(sub3.groupby('ETHRACE2A').std())

mc1 = multi.MultiComparison(sub3['NUMCIGMO_EST'], sub3['ETHRACE2A'])
print(mc1.tukeyhsd().summary())
