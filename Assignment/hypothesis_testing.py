from Assignment import visualisation_bivariate
from Assignment import helpers


dataset = visualisation_bivariate.dataset

"""
Hypothesis:
H0: The type of parents and whether their children have tried smoking cigarettes are independent (no association).
H1: The type of parents and whether their children have tried smoking cigarettes are dependent to each other.

Significance level: α = 0.05
"""
helpers.chi2test(dataset=dataset,
                 var_b='H1TO1',
                 var_a='PARENT_TYPES')
