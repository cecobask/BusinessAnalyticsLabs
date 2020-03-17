import pandas
import numpy
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Set pandas to show all columns and rows in data frame.
pandas.set_option('display.max_columns', None)
pandas.set_option('display.max_rows', None)

# Load data.
gapminder_data = pandas.read_csv('../gapminder.csv', low_memory=False)

# replace blanks
gapminder_data['lifeexpectancy'] = gapminder_data['lifeexpectancy'].replace(' ', numpy.NaN)
gapminder_data['employrate'] = gapminder_data['employrate'].replace(' ', numpy.NaN)

# numeric variables that are read into python from the csv file as strings (objects)
# with empty cells should be converted back to numeric format using convert_objects function
gapminder_data['lifeexpectancy'] = pandas.to_numeric(gapminder_data['lifeexpectancy'])
gapminder_data['employrate'] = pandas.to_numeric(gapminder_data['employrate'])

"""
Hypothesis:
H0: Life expectancy has no relationship with employ rate.
H1: There is a relationship between life expectancy and employ rate.
"""

# Subset of only two variables.
subset2 = gapminder_data[['employrate', 'lifeexpectancy']].dropna()

# Sample size.
print(len(subset2))

print("Mean - employrate")
mean_emp = subset2['employrate'].mean()
print(mean_emp)

print("STD - employrate")
std_emp = subset2['employrate'].std()
print(std_emp)

print("Mean - lifeexpectancy")
mean_life = subset2['employrate'].mean()
print(mean_life)

print("STD - lifeexpectancy")
std_life = subset2['employrate'].std()
print(std_life)

# regression for association between employ rate and life expectancy
print('OLS regression model for the association between life expectancy and employ rate')
reg1 = smf.ols('employrate ~ lifeexpectancy', data=gapminder_data).fit()
print(reg1.summary())


result = """
---------------------------------------------------------------------------------------

Number of observations:     176
p-value (lifeexpectancy):   1.0232954133794283e-05
R-squared:                  0.106

---------------------------------------------------------------------------------------

Since p-value is less than 0.05 (alpha value), we can reject the null hypothesis
and conclude that there is a relationship between life expectancy and employ rate.
We are 95% confident that null hypothesis is rejected.

The R-Squared value shows that the life expectancy variable accounts for 10% of the variance in employ rate variable.
The intercept is 82.6 and is statistically significant with a p-value of <.01.

---------------------------------------------------------------------------------------

If the life expectancy is age 80, employ rate would be calculated in the following way:

Ŷ = ß0 + ß1 * x
ß0 = 82.6
ß1 = -0.3
x = 80

Ŷ = 82.6 + (-0.3) * 80
Ŷ = 58.6

Results:    For life expectancy of age 80, the employ rate is 58.6
            For every one unit increase in x we would expect y to decrease by 0.3

---------------------------------------------------------------------------------------
"""

print(result)
