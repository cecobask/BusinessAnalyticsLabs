#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 20:09:23 2020
@author: bask
"""

import pandas

# Dictionary of columns to be used.
nesarc_dict = {
    'TAB12MDX': 'NICOTINE_DEPENDENCE',
    'CHECK321': 'SMOKING_STATUS',
    'S3AQ3B1': 'FREQUENCY_OF_SMOKING',
    'S3AQ3C1': 'QUANTITY_SMOKED',
    'S2AQ3': 'DRANK_ALCOHOL',
    'S2AQ8A': 'ALCOHOL_FREQUENCY'
}

# Load the dataset.
nesarc_data = pandas.read_csv('../nesarc_pds.csv', low_memory=False)

# Rename columns.
print('data read, performing rename operation')
nesarc_data.rename(columns=nesarc_dict, inplace=True)
print('data fetched')

# Test the outcome of renaming operation.
print(nesarc_data.columns)
print('counts for TAB12MDX - Nicotine dependence in the past 12 months, yes=1')
c1 = nesarc_data["NICOTINE_DEPENDENCE"].value_counts(sort=True)
print(c1)

print('old column amount: ' + str(len(nesarc_data.columns)))
# Update dataframe to only contain the columns from the dictionary.
nesarc_data = pandas.DataFrame(nesarc_data, columns=nesarc_dict.values())
print('new column amount: ' + str(len(nesarc_data.columns)))
