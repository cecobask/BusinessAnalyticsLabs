import pandas as pd
import numpy

# Load dataset.
addhealth_data = pd.read_csv('../addhealth_pds.csv', low_memory=False)

# Restrict dataset to observations that know their biological parents.
dataset = addhealth_data[(addhealth_data['H1NF1'] == 7) & (addhealth_data['H1NM1'] == 7)].copy()
pd.set_option("display.precision", 3)  # Set display results to 0 decimal points.

# Section 14: Resident Mother

dataset['H1RM1'] = dataset['H1RM1'].replace([11, 12, 96, 97, 98], numpy.nan)  # Replace unnecessary values with null.

# Section 15: Resident Father

dataset['H1RF1'] = dataset['H1RF1'].replace([11, 12, 96, 97, 98], numpy.nan)

# Section 16: Relations with Parents

# Loop over indices array and replace unnecessary values with null.
for i in [*range(1, 8), 9, 13, 10, 14]:
    dataset[f'H1WP{i}'] = dataset[f'H1WP{i}'].replace([6, 7, 8, 9], numpy.nan)


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


# Create a new variable using a subset of the original dataset.
dataset['PARENT_TYPES'] = dataset.loc[:, ['H1WP1', 'H1WP2', 'H1WP3', 'H1WP4', 'H1WP5', 'H1WP6', 'H1WP7']] \
    .apply(lambda row: parents_type(row), axis=1)

# Section 28: Tobacco, Alcohol, Drugs

dataset['H1TO1'] = dataset['H1TO1'].replace([6, 8, 9], numpy.nan)
dataset['H1TO2'] = dataset['H1TO2'].replace([0, 96, 97, 98], numpy.nan)
dataset['H1TO7'] = dataset['H1TO7'].replace([0, 96, 97, 98], numpy.nan)

# Create custom bins from variable dataset['H1TO2'].
dataset['H1TO2_BINS'] = pd.cut(dataset['H1TO2'],
                                 [0, 5, 10, 15, 20],
                                 labels=['1-5', '6-10', '11-15', '16-20'])

dataset['CIG_MONTHLY'] = dataset['H1TO7'] * 30.42  # Cigarettes per day * average number of days per month.
dataset['CIG_PACKS_MONTHLY'] = round(dataset['CIG_MONTHLY'] / 20)  # Typically a pack contains 20 cigarettes.
dataset['CIG_PACKS_MONTHLY_BINS'] = pd.cut(dataset['CIG_PACKS_MONTHLY'],  # Custom category bins.
                                          [0, 3, 6, 9, 136],
                                          labels=['1-3', '4-6', '7-9', '10+'])

"""
These printouts will be shown only when the script is ran.
Isolated them to prevent cluttering the console when importing this script for use in other scripts.
"""
if __name__ == '__main__':
    print(
        "\n==========================================================================================================\n"
        "Number of observations/rows in the AddHealth dataset:\n\n"
        f"{len(addhealth_data)}")

    print(
        "\n==========================================================================================================\n"
        "Number of variables/columns in the AddHealth dataset:\n\n"
        f"{len(addhealth_data.columns)}")

    # Section 12: Non-Resident Biological Mother

    print(
        "\n==========================================================================================================\n"
        "Number of people that know their mother:\n\n"
        f"{dataset['H1NM1'].value_counts().sum()}")

    # Section 13: Non-Resident Biological Father

    print(
        "\n==========================================================================================================\n"
        "Number of people that know their father:\n\n"
        f"{dataset['H1NF1'].value_counts().sum()}")

    # Section 14: Resident Mother

    print(
        "\n==========================================================================================================\n"
        "How far in school did the mother go? (%)\n"
        "#1 eighth grade or less\n"
        "#2 more than eighth grade, but did not graduate from high school\n"
        "#3 went to a business, trade, or vocational school instead of high school\n"
        "#4 high school graduate\n"
        "#5 completed a GED\n"
        "#6 went to a business, trade, or vocational school after high school\n"
        "#7 went to college, but did not graduate\n"
        "#8 graduated from a college or university\n"
        "#9 professional training beyond a four-year college or university\n"
        "#10 She never went to school.\n\n"
        f"{dataset['H1RM1'].value_counts(normalize=True)}")

    # Section 15: Resident Father

    print(
        "\n==========================================================================================================\n"
        "How far in school did the father go? (%))\n"
        "# Answer codes are identical to previous printout.\n\n"
        f"{dataset['H1RF1'].value_counts(normalize=True)}")

    # Section 16: Relations with Parents

    print(
        "\n==========================================================================================================\n"
        "How close do you feel to your mother? (%)\n"
        "#1 not at all\n"
        "#2 very little\n"
        "#3 somewhat\n"
        "#4 quite a bit\n"
        "#5 very much\n\n"
        f"{dataset['H1WP9'].value_counts(normalize=True)}")

    print(
        "\n==========================================================================================================\n"
        "How close do you feel to your father? (%)\n"
        "# Answer codes are identical to previous printout.\n\n"
        f"{dataset['H1WP13'].value_counts(normalize=True)}")

    print(
        "\n==========================================================================================================\n"
        "How much do you think your mother cares about you? (%)\n"
        "# Answer codes are identical to previous printout.\n\n"
        f"{dataset['H1WP10'].value_counts(normalize=True)}")

    print(
        "\n==========================================================================================================\n"
        "How much do you think your father cares about you? (%)\n"
        "# Answer codes are identical to previous printout.\n\n"
        f"{dataset['H1WP14'].value_counts(normalize=True)}")

    print(
        "\n==========================================================================================================\n"
        "Ratio of bossy to soft parents (%):\n\n"
        f"{dataset['PARENT_TYPES'].value_counts(normalize=True)}")

    # Section 28: Tobacco, Alcohol, Drugs

    print(
        "\n==========================================================================================================\n"
        "Have you ever tried cigarette smoking, even just 1 or 2 puffs? (%)\n"
        "#0 no\n"
        "#1 yes\n\n"
        f"{dataset['H1TO1'].value_counts(normalize=True)}")

    print(
        "\n==========================================================================================================\n"
        "How old were you when you smoked a whole cigarette for the first time? (%)\n"
        "This is a newly created variable that uses 'pandas.cut()' function to create custom age bins.\n\n"
        f"{dataset['H1TO2_BINS'].value_counts(normalize=True, sort=False)}")

    print(
        "\n==========================================================================================================\n"
        "Descriptive statistics about the age of smokers' when they first tried smoking cigarettes:\n\n"
        f"{dataset['H1TO2'].describe()}")

    print(
        "\n==========================================================================================================\n"
        "Bins of cigarette packs smoked per month (%):\n"
        "This is a newly created variable that uses 'pandas.cut()' function to create custom age bins.\n\n"
        f"{dataset['CIG_PACKS_MONTHLY_BINS'].value_counts(sort=False, normalize=True)}")

    print(
        "\n==========================================================================================================\n"
        "Descriptive statistics about number of cigarettes smoked by smokers per month:\n\n"
        f"{dataset['CIG_MONTHLY'].describe()}")
