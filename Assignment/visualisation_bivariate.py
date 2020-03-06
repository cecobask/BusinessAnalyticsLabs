from Assignment import visualisation_univariate
from Assignment import helpers
import pandas as pd
import seaborn
import matplotlib.pyplot as plt

# Modified dataset from visualisation_univariate script.
dataset = visualisation_univariate.dataset

# Graph showing the relationship between the types of parents
# and their children's mean smoking attempt rate.
helpers.build_catplot(data=dataset,
                      x='PARENT_TYPES',
                      y='H1TO1',
                      title='RELATIONSHIP BETWEEN THE TYPES OF PARENTS AND\n'
                            'THEIR CHILDREN\'S MEAN SMOKING ATTEMPT RATE',
                      xlabel='PARENT TYPES',
                      ylabel='MEAN SMOKING ATTEMPT RATE')

# Graph showing the relationship between the age when children first smoked a cigarette
# and the number of cigarette packs they smoked per month. (Aged 10 or over)
plt.figure(figsize=(7.5, 4.8))
seaborn.regplot(data=dataset,
                x="H1TO2",
                y="CIG_PACKS_MONTHLY",
                fit_reg=True)
plt.xlabel('CHILDREN\'S AGE')
plt.ylabel('CIGARETTE PACKS MONTHLY')
plt.title('RELATIONSHIP BETWEEN THE AGE WHEN CHILDREN FIRST SMOKED A CIGARETTE\n'
          'AND THE NUMBER OF CIGARETTE PACKS THEY SMOKE PER MONTH')
plt.show()

# Graph showing the relationship between the education level of children's parents
# and age at which children smoked their first cigarette.
helpers.build_catplot(data=dataset,
                      x='H1TO2_BINS',
                      y='PARENTS_EDU_LEVEL',
                      title='RELATIONSHIP BETWEEN THE EDUCATION LEVEL OF CHILDREN\'S PARENTS\n'
                            'AND AGE AT WHICH CHILDREN SMOKED THEIR FIRST CIGARETTE',
                      xlabel='FIRST CIGARETTE AT AGE',
                      ylabel='MEAN PARENTS EDUCATION LEVEL')

# Bin into 2 categories - bond or no bond.
dataset['PARENTS_CHILD_BOND_OR_NOT'] = pd.cut(dataset['PARENTS_CHILD_BOND'],
                                              [0, 2.5, 5],
                                              labels=['NO', 'YES'])

# Calculate yearly amount of cigarette packs smoked by participants.
dataset['CIG_PACKS_YEARLY'] = round(dataset['CIG_MONTHLY'] / 12)

# Graph showing the relationship between bond of children with their parents and
# the number of cigarette packs children smoke per year.
helpers.build_catplot(data=dataset,
                      x='PARENTS_CHILD_BOND_OR_NOT',
                      y='CIG_PACKS_YEARLY',
                      title='RELATIONSHIP BETWEEN BOND OF CHILDREN WITH THEIR PARENTS AND\n'
                            'THE NUMBER OF CIGARETTE PACKS CHILDREN SMOKE PER YEAR',
                      xlabel='CHILDREN:PARENTS BOND',
                      ylabel='MEAN CIGARETTE PACKS YEARLY')


"""
These printouts will be shown only when the script is ran.
Isolated them to prevent cluttering the console when importing this script for use in other scripts.
"""
if __name__ == '__main__':
    print(
        "\n==========================================================================================================\n"
        "Descriptive statistics about the bond score between parents and children:\n\n"
        f"{dataset['PARENTS_CHILD_BOND'].describe()}")

    print(
        "\n==========================================================================================================\n"
        "Descriptive statistics about the number of cigarette packs smoked by children per month:\n\n"
        f"{dataset['CIG_PACKS_YEARLY'].describe()}")
