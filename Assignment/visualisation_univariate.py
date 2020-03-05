from Assignment import initial_analysis
from Assignment import helpers
import pandas as pd

dataset = initial_analysis.dataset  # Modified dataset from initial_analysis script.

# Make a subset that includes ['H1RF1', 'H1RM1'] columns and apply a function to each row.
dataset['PARENTS_EDU_LEVEL'] = dataset.loc[:, ['H1RF1', 'H1RM1']] \
    .apply(lambda row: helpers.parents_edu_level(row), axis=1)

# Bin into education levels.
dataset['PARENTS_EDU_LEVEL_BINS'] = pd.cut(dataset['PARENTS_EDU_LEVEL'],
                                           [0, 4, 6, 8, 9, 10],
                                           labels=['High-school', 'Vocational', 'Uni', 'Beyond Uni', 'None'])

# Make a subset that includes ['H1WP9', 'H1WP10', 'H1WP13', 'H1WP14'] columns and apply a function to each row.
dataset['PARENTS_CHILD_BOND'] = dataset.loc[:, ['H1WP9', 'H1WP10', 'H1WP13', 'H1WP14']] \
    .apply(lambda row: helpers.parents_child_bond(row), axis=1)

# Bin into bonding levels.
dataset['PARENTS_CHILD_BOND_BINS'] = pd.cut(dataset['PARENTS_CHILD_BOND'],
                                            [0, 2, 3.5, 5],
                                            labels=['Low', 'Medium', 'High'])

# Rename values of 'H1TO1' column.
dataset['H1TO1RN'] = dataset.loc[:, 'H1TO1'] \
    .apply(lambda row: helpers.rename_h1to1(row))


"""
These graphs will be shown only when the script is ran.
Isolated them to prevent cluttering the view when importing this script for use in other scripts.
"""
if __name__ == '__main__':
    # Display a chart showcasing the frequency distribution and percentages
    # of parents' education level.
    helpers.build_countplot(dataset=dataset,
                            column_name='PARENTS_EDU_LEVEL_BINS',
                            title='FREQUENCY DISTRIBUTION AND PERCENTAGES FOR THE AVERAGE\n'
                                  'PARENTS EDUCATION LEVEL OF KIDS IN THE ADD HEALTH STUDY',
                            ylabel='EDUCATION LEVEL')

    # Display a chart showcasing the frequency distribution and percentages
    # about the kid - parents bonding levels.
    helpers.build_countplot(dataset=dataset,
                            column_name='PARENTS_CHILD_BOND_BINS',
                            title='FREQUENCY DISTRIBUTION AND PERCENTAGES FOR BONDING LEVELS\n'
                                  'BASED ON THE RELATIONSHIPS PARENTS-CHILD AND CHILD-PARENTS',
                            ylabel='BONDING LEVEL')

    # Display a chart showcasing the frequency distribution and percentages
    # about the ratio of soft to bossy parent types.
    helpers.build_countplot(dataset=dataset,
                            column_name='PARENT_TYPES',
                            title='FREQUENCY DISTRIBUTION AND PERCENTAGES FOR\nTHE RATIO OF BOSSY TO SOFT PARENT TYPES',
                            ylabel='PARENTS TYPE')

    # Display a chart showcasing the frequency distribution and percentages
    # whether the children have ever tried smoking cigarettes.
    helpers.build_countplot(dataset=dataset,
                            column_name='H1TO1RN',
                            title='FREQUENCY DISTRIBUTION AND PERCENTAGES OF RESPONSE TO QUESTION:\n'
                                  'HAS THE CHILD EVER TRIED SMOKING CIGARETTES?',
                            ylabel='ANSWER')

    # Display a chart showcasing the frequency distribution and percentages
    # of the age at which children smoked their first cigarette.
    helpers.build_countplot(dataset=dataset,
                            column_name='H1TO2_BINS',
                            title='FREQUENCY DISTRIBUTION AND PERCENTAGES OF THE AGE AT WHICH\n'
                                  'CHILDREN HAVE SMOKED THEIR FIRST CIGARETTE',
                            ylabel='AGE')

    # Display a chart showcasing the frequency distribution and percentages
    # of cigarette packs smoked per month by children that are smokers
    helpers.build_countplot(dataset=dataset,
                            column_name='CIG_PACKS_MONTHLY_BINS',
                            title='FREQUENCY DISTRIBUTION AND PERCENTAGES OF CIGARETTE PACKS\n'
                                  'SMOKED PER MONTH BY CHILDREN THAT ARE SMOKERS',
                            ylabel='CIGARETTE PACKS')
