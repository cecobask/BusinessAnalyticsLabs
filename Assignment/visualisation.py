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


# Make a subset that includes ['H1RF1', 'H1RM1'] columns and apply a function to each row.
dataset['PARENTS_EDU_LEVEL'] = dataset.loc[:, ['H1RF1', 'H1RM1']] \
    .apply(lambda row: parents_edu_level(row), axis=1)

# Bin into education levels.
dataset['PARENTS_EDU_LEVEL_BINS'] = pd.cut(dataset['PARENTS_EDU_LEVEL'],
                                           [0, 4, 6, 8, 9, 10],
                                           labels=['High-school', 'Vocational', 'Uni', 'Beyond Uni', 'None'])

# Display a chart showcasing the education level of parents.
seaborn.countplot(x='PARENTS_EDU_LEVEL_BINS', data=dataset)
plt.xlabel('EDUCATION LEVEL')
plt.ylabel('FREQUENCY')
plt.title('AVERAGE PARENTS EDUCATION LEVEL OF\nKIDS IN THE ADD HEALTH STUDY', wrap=True)
plt.show()


def parents_child_bond(row):
    """
    Calculates how close children are with their parents and how much they think their parents care about them.
    The results of these two calculations is used to determine a bond score.
    Values indicate:
    #1 not at all
    #2 very little
    #3 somewhat
    #4 quite a bit
    #5 very much
    :param row: Series
    :return: numpy.float64
    """
    mother2child = row['H1WP10']
    father2child = row['H1WP14']
    child2mother = row['H1WP9']
    child2father = row['H1WP13']

    # Calculate individual affinities first.
    parents2child = (mother2child + father2child) / 2
    child2parents = (child2mother + child2father) / 2

    return (parents2child + child2parents) / 2


# Make a subset that includes ['H1WP9', 'H1WP10', 'H1WP13', 'H1WP14'] columns and apply a function to each row.
dataset['PARENTS_CHILD_BOND'] = dataset.loc[:, ['H1WP9', 'H1WP10', 'H1WP13', 'H1WP14']] \
    .apply(lambda row: parents_child_bond(row), axis=1)

# Bin into bonding levels.
dataset['PARENTS_CHILD_BOND_BINS'] = pd.cut(dataset['PARENTS_CHILD_BOND'],
                                            [0, 2, 3.5, 5],
                                            labels=['Low', 'Medium', 'High'])

# Display a chart showcasing the bonding levels.
seaborn.countplot(x='PARENTS_CHILD_BOND_BINS', data=dataset)
plt.xlabel('BONDING LEVEL')
plt.ylabel('FREQUENCY')
plt.title('BONDING LEVELS BASED ON THE RELATIONSHIPS\nPARENTS-CHILD AND CHILD-PARENTS', wrap=True)
plt.show()
