from Assignment import initial_analysis
import seaborn
import matplotlib.pyplot as plt
import pandas as pd

dataset = initial_analysis.dataset  # Modified dataset from initial_analysis script.


def parents_edu_level(row):
    """
    Determines the average education level of the parents.
    :param row: Series
    :return: numpy.float64
    """
    mother = row['H1RM1']
    father = row['H1RF1']

    return (mother + father) / 2


# Make a subset that includes just 'H1RF1', 'H1RM1' variables and apply a function to each row.
dataset['PARENTS_EDU_LEVEL'] = dataset.loc[:, ['H1RF1', 'H1RM1']]\
    .apply(lambda row: parents_edu_level(row), axis=1)
dataset['PARENTS_EDU_LEVEL_BINS'] = pd.cut(dataset['PARENTS_EDU_LEVEL'],  # Bin into education level
                                           [0, 4, 6, 8, 9, 10],
                                           labels=['High-school', 'Vocational', 'Uni', 'Beyond Uni', 'None'])

# Display charts using matplotlib and seaborn.
seaborn.countplot(x='PARENTS_EDU_LEVEL_BINS', data=dataset)
plt.xlabel('EDUCATION LEVEL')
plt.ylabel('FREQUENCY')
plt.title('AVERAGE EDUCATION LEVEL OF PARENTS', wrap=True)
plt.show()
