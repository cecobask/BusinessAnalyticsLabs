from Assignment import initial_analysis
from Assignment import helpers
from Assignment import visualisation_univariate
import pandas as pd
import seaborn
import matplotlib.pyplot as plt

dataset = visualisation_univariate.dataset  # Modified dataset from visualisation_univariate script.

# Graph showing the relationship between the types of parents and their children's mean smoking attempt rate.
seaborn.catplot(x='PARENT_TYPES', y='H1TO1', data=dataset, kind='bar', ci=None)
plt.title('RELATIONSHIP BETWEEN THE TYPES OF PARENTS AND THEIR CHILDRENS\'S MEAN SMOKING ATTEMPT RATE')
plt.xlabel('PARENT TYPES')
plt.ylabel('SMOKING ATTEMPT MEAN RATE')
plt.show()

# Graph showing the relationship between the age when children first smoked a cigarette
# and the number of cigarette packs they smoked per month.
plt.figure(figsize=(7.5, 4.8))
seaborn.regplot(y="CIG_PACKS_MONTHLY", x="H1TO2", data=dataset, fit_reg=True)
plt.xlabel('CHILDREN\'S AGE')
plt.ylabel('CIGARETTE PACKS MONTHLY')
plt.title('RELATIONSHIP BETWEEN THE AGE WHEN CHILDREN FIRST SMOKED A CIGARETTE\n'
          'AND THE NUMBER OF CIGARETTE PACKS THEY SMOKE PER MONTH')
plt.show()
