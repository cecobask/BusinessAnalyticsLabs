from Assignment import visualisation_bivariate
from Assignment import helpers
import pandas as pd

dataset = visualisation_bivariate.dataset

"""
Hypothesis:
H0: The type of parents and whether their children have tried smoking cigarettes are independent (no association).
H1: The type of parents and whether their children have tried smoking cigarettes are dependent to each other.

Significance level: α = 0.05
"""
helpers.chi2test(dataset=dataset,
                 var_b='H1TO1',
                 var_a='PARENT_TYPES',
                 h0='The type of parents and whether their children have tried '
                    'smoking cigarettes are independent (no association).',
                 h1='The type of parents and whether their children have tried '
                    'smoking cigarettes are dependent to each other.')

f"""
Hypothesis:
H0: There is no relationship between the number of cigarette packs smoked per month
    between children aged under 13 and over 13.
H1: There is a relationship between the number of cigarette packs smoked per month
    between children aged under 13 and over 13.

Significance level: α = 0.05
"""

# Subset of only two variables.
subset = dataset.loc[:, ['CIG_PACKS_MONTHLY', 'H1TO2']]
subset.rename(columns={"H1TO2": "SMOKE_AGES"}, inplace=True)  # Rename column.

# Bin into 2 categories - younger and older than 13 years.
subset['SMOKE_AGES_CAT'] = pd.cut(subset['SMOKE_AGES'],
                                  [0, 12, 18],
                                  labels=['<13', '>=13'])
helpers.anova(dataset=subset,
              var_a='CIG_PACKS_MONTHLY',
              var_b='SMOKE_AGES_CAT',
              h0='There is no relationship between the number of cigarette packs smoked per month'
                 'between children aged under 13 and over 13.',
              h1='There is a relationship between the number of cigarette packs smoked per month'
                 'between children aged under 13 and over 13.')
