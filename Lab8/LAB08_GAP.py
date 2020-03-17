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
gapminder_data['internetuserate'] = gapminder_data['internetuserate'].replace(' ', numpy.NaN)
gapminder_data['urbanrate'] = gapminder_data['urbanrate'].replace(' ', numpy.NaN)

# numeric variables that are read into python from the csv file as strings (objects)
# with empty cells should be converted back to numeric format using convert_objects function
gapminder_data['urbanrate'] = pandas.to_numeric(gapminder_data['urbanrate'])
gapminder_data['internetuserate'] = pandas.to_numeric(gapminder_data['internetuserate'])

# regression for association between urbandrate and internet use rate
print('OLS regression model for the association between urbanrate and internet use rate')
reg1 = smf.ols('internetuserate ~ urbanrate', data=gapminder_data).fit()
print(reg1.summary())
