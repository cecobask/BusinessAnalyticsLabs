import pandas
import numpy
import seaborn
import matplotlib.pyplot as plt

# Set pandas to show all columns and rows in data frame.
pandas.set_option('display.max_columns', None)
pandas.set_option('display.max_rows', None)

# Load data.
gapminder_data = pandas.read_csv('../gapminder.csv', low_memory=False)
subset1 = gapminder_data.copy()

# Remove null values and convert variables to numeric.
vars_to_convert = ['urbanrate', 'internetuserate', 'incomeperperson', 'hivrate']
for var in vars_to_convert:
    subset1[var] = subset1[var].replace(' ', numpy.nan)
    subset1[var] = pandas.to_numeric(subset1[var])
    print(f"\nDescriptive statistics for {var}")
    print(subset1[var].describe())

# Basic scatter plot.
seaborn.regplot(x="urbanrate", y="internetuserate", fit_reg=True, data=subset1)
plt.xlabel('Urban Rate')
plt.ylabel('Internet Use Rate')
plt.title('Scatter plot for the association between Urban Rate and Internet Use Rate')
plt.show()

# Income per Person : Internet Use Rate scatter plot.
plt.figure(figsize=(9, 9))  # change size of chart.
seaborn.regplot(x="incomeperperson", y="internetuserate", fit_reg=False, data=subset1)
plt.xlabel('Income per Person')
plt.ylabel('Internet Use Rate')
plt.title('Scatter plot for the association between Income per Person and Internet Use Rate')
plt.show()

# HIV Rate : Internet Use Rate scatter plot.
seaborn.regplot(x="incomeperperson", y="hivrate", fit_reg=True, data=subset1)
plt.xlabel('Income per Person')
plt.ylabel('HIV Rate')
plt.title('Scatter plot for the association between Income per Person and HIV Rate')
plt.show()

# Previous plot did not help determine the relationship
# Hence, convert 'incomeperperson' into categorical variable.
subset1['incomegrp4'] = pandas.qcut(subset1['incomeperperson'], 4, labels=[
    '1=25%tile', '2=50%tile', '3=75%tile', '4=100%tile'
])
print(subset1['incomegrp4'].value_counts(sort=False))

# Bivariate bar graph - Categorical:Quantitative.
seaborn.catplot(x='incomegrp4', y='hivrate', data=subset1, kind='bar', ci=None)
plt.xlabel('Income Group')
plt.ylabel('Mean HIV Rate')
plt.show()
