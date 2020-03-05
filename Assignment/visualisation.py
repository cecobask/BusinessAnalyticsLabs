from Assignment import initial_analysis
import seaborn
import matplotlib.pyplot as plt
import pandas as pd

dataset = initial_analysis.dataset  # Modified dataset from initial_analysis script.


def show_axis_percentages(plot, column):
    """
    Helper function that adds percentages to the right of horizontal plot bars.
    :param plot: AxesSubplot
    :param column: Series
    :return: None
    """
    for p in plot.patches:
        percentage = '{:.1f}%'.format(100 * p.get_width() / column.value_counts().sum())
        x = p.get_x() + p.get_width() + 0.02
        y = p.get_y() + p.get_height() / 2
        plot.annotate(percentage, (x, y))


def parents_edu_level(row):
    """
    Determines the average education level of the parents.
    :param row: Series
    :return: numpy.float64
    """
    mother = row['H1RM1']
    father = row['H1RF1']

    return (mother + father) / 2


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


def rename_h1to1(row):
    """
    Replaces floats with readable string values for display purposes.
    :param row: float64
    :return: str
    """
    if row == 0:
        return 'NO'
    else:
        return 'YES'


# Make a subset that includes ['H1RF1', 'H1RM1'] columns and apply a function to each row.
dataset['PARENTS_EDU_LEVEL'] = dataset.loc[:, ['H1RF1', 'H1RM1']] \
    .apply(lambda row: parents_edu_level(row), axis=1)

# Bin into education levels.
dataset['PARENTS_EDU_LEVEL_BINS'] = pd.cut(dataset['PARENTS_EDU_LEVEL'],
                                           [0, 4, 6, 8, 9, 10],
                                           labels=['High-school', 'Vocational', 'Uni', 'Beyond Uni', 'None'])

# Display a chart showcasing the education level of parents.
plt.figure(figsize=(7.5, 4.8))  # Change size of the chart.
ax = seaborn.countplot(y='PARENTS_EDU_LEVEL_BINS', data=dataset)
plt.ylabel('EDUCATION LEVEL')
plt.xlabel('FREQUENCY')
plt.title('FREQUENCY DISTRIBUTION AND PERCENTAGES FOR THE AVERAGE\n'
          'PARENTS EDUCATION LEVEL OF KIDS IN THE ADD HEALTH STUDY')
show_axis_percentages(ax, dataset['PARENTS_EDU_LEVEL_BINS'])
plt.show()

# Make a subset that includes ['H1WP9', 'H1WP10', 'H1WP13', 'H1WP14'] columns and apply a function to each row.
dataset['PARENTS_CHILD_BOND'] = dataset.loc[:, ['H1WP9', 'H1WP10', 'H1WP13', 'H1WP14']] \
    .apply(lambda row: parents_child_bond(row), axis=1)

# Bin into bonding levels.
dataset['PARENTS_CHILD_BOND_BINS'] = pd.cut(dataset['PARENTS_CHILD_BOND'],
                                            [0, 2, 3.5, 5],
                                            labels=['Low', 'Medium', 'High'])

# Display a chart showcasing the frequency distribution and percentages
# about the kid - parents bonding levels.
plt.figure(figsize=(7.5, 4.8))  # Change size of the chart.
ax = seaborn.countplot(y='PARENTS_CHILD_BOND_BINS', data=dataset)
plt.ylabel('BONDING LEVEL')
plt.xlabel('FREQUENCY')
plt.title('FREQUENCY DISTRIBUTION AND PERCENTAGES FOR BONDING LEVELS\n'
          'BASED ON THE RELATIONSHIPS PARENTS-CHILD AND CHILD-PARENTS')
show_axis_percentages(ax, dataset['PARENTS_CHILD_BOND_BINS'])
plt.show()

# Display a chart showcasing the frequency distribution and percentages
# about the ratio of soft to bossy parent types.
plt.figure(figsize=(7.5, 4.8))  # Change size of the chart.
ax = seaborn.countplot(y='PARENT_TYPES', data=dataset)
plt.title('FREQUENCY DISTRIBUTION AND PERCENTAGES ABOUT\nTHE RATIO OF BOSSY TO SOFT PARENT TYPES')
plt.ylabel('PARENTS TYPE')
plt.xlabel('FREQUENCY')
show_axis_percentages(ax, dataset['PARENT_TYPES'])
plt.show()

# Rename values of 'H1TO1' column.
dataset['H1TO1RN'] = dataset.loc[:, 'H1TO1'] \
    .apply(lambda row: rename_h1to1(row))

# Display a chart showcasing the frequency distribution and percentages
# whether the children have ever tried smoking cigarettes.
plt.figure(figsize=(7.5, 4.8))  # Change size of the chart.
ax = seaborn.countplot(y='H1TO1RN', data=dataset)
plt.title('FREQUENCY DISTRIBUTION AND PERCENTAGES OF RESPONSE TO QUESTION:\n'
          'HAS THE CHILD EVER TRIED SMOKING CIGARETTES?')
plt.ylabel('ANSWER')
plt.xlabel('FREQUENCY')
show_axis_percentages(ax, dataset['H1TO1RN'])
plt.show()

# Display a chart showcasing the frequency distribution and percentages
# of the age at which children smoked their first cigarette.
plt.figure(figsize=(7.5, 4.8))  # Change size of the chart.
ax = seaborn.countplot(y='H1TO2_BINS', data=dataset)
plt.title('FREQUENCY DISTRIBUTION AND PERCENTAGES OF THE AGE AT WHICH\n'
          'CHILDREN HAVE SMOKED THEIR FIRST CIGARETTE')
plt.ylabel('AGE')
plt.xlabel('FREQUENCY')
show_axis_percentages(ax, dataset['H1TO2_BINS'])
plt.show()

# Display a chart showcasing the frequency distribution and percentages
# of cigarette packs smoked per month by children that are smokers
plt.figure(figsize=(7.5, 4.8))  # Change size of the chart.
ax = seaborn.countplot(y='CIG_PACKS_MONTHLY_BINS', data=dataset)
plt.title('FREQUENCY DISTRIBUTION AND PERCENTAGES OF CIGARETTE PACKS\n'
          'SMOKED PER MONTH BY CHILDREN THAT ARE SMOKERS')
plt.ylabel('CIGARETTE PACKS')
plt.xlabel('FREQUENCY')
show_axis_percentages(ax, dataset['CIG_PACKS_MONTHLY_BINS'])
plt.show()
