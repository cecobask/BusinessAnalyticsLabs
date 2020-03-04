import pandas as pd
import numpy

# Load dataset.
addhealth_data = pd.read_csv('../addhealth_pds.csv', low_memory=False)

print(f"""==============================================================================================================
Number of observations/rows in the AddHealth dataset:\n
{len(addhealth_data)}
""")

print(f"""==============================================================================================================
Number of variables/columns in the AddHealth dataset:\n
{len(addhealth_data.columns)}
""")

# Restrict dataset to observations that know their parents.
dataset = addhealth_data[(addhealth_data['H1NF1'] == 7) & (addhealth_data['H1NM1'] == 7)].copy()
# Set display results to 0 decimal points.
pd.set_option("display.precision", 3)

# Section 12: Non-Resident Biological Mother

print(f"""==============================================================================================================
Number of people that know their mother:\n
{dataset['H1NM1'].value_counts().sum()}
""")

# Section 13: Non-Resident Biological Father

print(f"""==============================================================================================================
Number of people that know their father:\n
{dataset['H1NF1'].value_counts().sum()}
""")

# Section 14: Resident Mother

dataset['H1RM1'] = dataset['H1RM1'].replace([11, 12, 96, 97, 98], numpy.nan)  # Replace unnecessary values with null.
print(f"""==============================================================================================================
How far in school did the mother go? (%)
#1 eighth grade or less
#2 more than eighth grade, but did not graduate from high school
#3 went to a business, trade, or vocational school instead of high school
#4 high school graduate
#5 completed a GED
#6 went to a business, trade, or vocational school after high school
#7 went to college, but did not graduate
#8 graduated from a college or university
#9 professional training beyond a four-year college or university
#10 She never went to school.\n
{dataset['H1RM1'].value_counts(normalize=True)}
""")

# Section 15: Resident Father

dataset['H1RF1'] = dataset['H1RF1'].replace([11, 12, 96, 97, 98], numpy.nan)
print(f"""==============================================================================================================
How far in school did the father go? (%)")
# Answer codes are identical to previous printout.\n
{dataset['H1RF1'].value_counts(normalize=True)}
""")

# Section 16: Relations with Parents

# Loop over indices array and replace unnecessary values with null.
for i in [9, 13, 10, 14]:
    dataset[f'H1WP{i}'] = dataset[f'H1WP{i}'].replace([6, 7, 8], numpy.nan)

print(f"""==============================================================================================================
How close do you feel to your mother? (%)
#1 not at all
#2 very little
#3 somewhat
#4 quite a bit
#5 very much\n
{dataset['H1WP9'].value_counts(normalize=True)}
""")

print(f"""==============================================================================================================
How close do you feel to your father? (%)
# Answer codes are identical to previous printout.\n
{dataset['H1WP13'].value_counts(normalize=True)}
""")

print(f"""==============================================================================================================
How much do you think your mother cares about you? (%)
# Answer codes are identical to previous printout.\n
{dataset['H1WP10'].value_counts(normalize=True)}
""")

print(f"""==============================================================================================================
How much do you think your father cares about you? (%)
# Answer codes are identical to previous printout.\n
{dataset['H1WP14'].value_counts(normalize=True)}
""")

# Loop over indices array and replace unnecessary values with null.
for i in range(1, 8):
    dataset[f'H1WP{i}'] = dataset[f'H1WP{i}'].replace([6, 7, 8, 9], numpy.nan)


def parents_type(row):
    """
    Determine if the parents are bossy or soft.
    The questions asked are of format: Do your parents let you make your own decisions about...
    Possible answers are:
    0 -- no
    1 -- yes
    :param row: Series
    :return: bool
    """
    return 'BOSSY' if 0 in row.values else 'SOFT'


# Create a new variable using a subset of the original dataset.
dataset['PARENTS_TYPES'] = dataset.loc[:, ['H1WP1', 'H1WP2', 'H1WP3', 'H1WP4', 'H1WP5', 'H1WP6', 'H1WP7']] \
    .apply(lambda row: parents_type(row), axis=1)
print(f"""==============================================================================================================
Ratio of bossy to soft parents (%):\n
{dataset['PARENTS_TYPES'].value_counts(normalize=True)}
""")

# Section 28: Tobacco, Alcohol, Drugs
# 1-8, 50

dataset['H1TO1'] = dataset['H1TO1'].replace([6, 8, 9], numpy.nan)
dataset['H1TO2'] = dataset['H1TO2'].replace([0, 96, 97, 98], numpy.nan)
dataset['H1TO7'] = dataset['H1TO7'].replace([0, 96, 97, 98], numpy.nan)

print(f"""==============================================================================================================
Have you ever tried cigarette smoking, even just 1 or 2 puffs? (%)
#0 no
#1 yes\n
{dataset['H1TO1'].value_counts(normalize=True)}
""")

# Create custom bins from variable dataset['H1TO2'].
dataset['H1TO2_BINNED'] = pd.cut(dataset['H1TO2'],
                                 [0, 5, 10, 15, 20],
                                 labels=['1-5', '6-10', '11-15', '16-20'])

print(f"""==============================================================================================================
How old were you when you smoked a whole cigarette for the first time? (%)
This is a newly created variable that uses 'pandas.cut()' function to create custom age bins.\n
{dataset['H1TO2_BINNED'].value_counts(normalize=True, sort=False)}
""")

print(f"""==============================================================================================================
Descriptive statistics about the age of smokers' when they first tried smoking cigarettes:\n
{dataset['H1TO2'].describe()}""")

dataset['CIG_MONTHLY'] = dataset['H1TO7'] * 30.42  # Cigarettes per day * average number of days per month.
dataset['CIG_PACKS_MONTHLY'] = round(dataset['CIG_MONTHLY'] / 20)  # Typically a pack contains 20 cigarettes.
dataset['CIG_PACKS_MONTHLY_CUT'] = pd.cut(dataset['CIG_PACKS_MONTHLY'],  # Custom category bins.
                                          [0, 3, 6, 9, 136],
                                          labels=['1-3', '4-6', '7-9', '10+'])

print(f"""==============================================================================================================
Number of cigarette packs smoked per month (%):\n
{dataset['CIG_PACKS_MONTHLY_CUT'].value_counts(sort=False, normalize=True)}""")

print(f"""==============================================================================================================
Descriptive statistics about number of cigarettes smoked by smokers per month:\n
{dataset['CIG_MONTHLY'].describe()}""")
