import seaborn
import matplotlib.pyplot as plt
import numpy
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


def parents_type(row):
    """
    Determine if the parents are bossy or soft.
    The questions asked are of format: Do your parents let you make your own decisions about...
    Possible answers are:
    0 -- no
    1 -- yes
    A parent is considered soft if they let their child make their own decisions about 4 or more questions (out of 7).
    :param row: Series
    :return: bool
    """
    # Create a dictionary with unique values (1 and 0) and their counts.
    unique, counts = numpy.unique(row.values, return_counts=True)
    counts_dict = dict(zip(unique, counts))
    yes_answers = counts_dict.get(1, 0)  # Get the number of 'yes' answers or replace with 0 if missing.
    return 'Soft' if yes_answers > 4 else 'Bossy'


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


def build_countplot(dataset, column_name, title, ylabel, xlabel='FREQUENCY'):
    """
    Abstract the creation and showing of countplot, as it is heavily used
    for my research questions, due to their categorical nature.
    :param dataset: DataFrame
    :param column_name: str
    :param title: str
    :param ylabel: str
    :param xlabel: str
    :return: None
    """
    plt.figure(figsize=(7.5, 4.8))
    ax = seaborn.countplot(y=column_name, data=dataset)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    show_axis_percentages(ax, dataset[column_name])
    plt.show()


def build_catplot(data, x, y, title, xlabel, ylabel, kind='bar', ci=None, height=6.3):
    """
    Abstract the creation and showing of catplot, as it is heavily used.
    :param data: DataFrame
    :param x: str
    :param y: str
    :param title: str
    :param xlabel: str
    :param ylabel: str
    :param kind: str
    :param ci: str
    :param height: float
    :return: None
    """
    seaborn.catplot(x=x, y=y, data=data, kind=kind, ci=ci, height=height)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def chi2test(dataset, var_a, var_b, h0, h1, alpha=0.05):
    """
    Creates a contingency table for passed in variables and
    runs chi-squared test to determine whether to reject/keep null hypothesis.
    :param dataset: DataFrame
    :param var_a: str
    :param var_b: str
    :param h0: str
    :param h1: str
    :param alpha: float
    :return: None
    """
    print('\n=======================================================================================================\n')

    # Create contingency table for passed in variables.
    crosstab = pd.crosstab(dataset[var_a], dataset[var_b])
    print(f'\n{crosstab}\n')

    # Chi-square test of independence of variables in a contingency table.
    print(f'\nRunning chi-squared test on variables \'{var_a}\' and \'{var_b}\':\n\n')
    stat, p, dof, expected = stats.chi2_contingency(crosstab)

    # Determine whether to reject or keep null hypothesis
    print(f'Significance: α = {alpha}\n'
          f'p-value: {p}\n'
          f'Degrees of freedom: {dof}\n'
          f'Expected: {expected}\n\n')

    if p <= alpha:
        print(f'Rejected H0. \n{h0}\n')
    else:
        print(f'Failed to reject H0. \n{h1}\n')


def anova(dataset, var_a, var_b, h0, h1, alpha=0.05):
    """
    Creates an OLS model model and determines whether
    to reject/keep null hypothesis based on the p-value.
    :param dataset: DataFrame
    :param var_a: str
    :param var_b: str
    :param h0: str
    :param h1: str
    :param alpha: float
    :return: None
    """
    # Using OLS function for calculating the F-statistic and associated p-value.
    model = smf.ols(formula=f'{var_a} ~ C({var_b})', data=dataset).fit()
    print(model.summary())

    p = model.pvalues[1]

    if p <= alpha:
        print(f'Rejected H0. \n{h0}\n')
    else:
        print(f'Failed to reject H0. \n{h1}\n')

    print(
        "\n==========================================================================================================\n"
        f'Means of {var_a} for all {var_b} categories:\n\n'
        f'{dataset.groupby(var_b).mean()}\n')
    print(
        "\n==========================================================================================================\n"
        f'Standard deviations of {var_a} for all {var_b} categories:\n\n'
        f'{dataset.groupby(var_b).std()}\n')
