# -*- coding: utf-8 -*-
"""
Data cleaning - Remove any examples where the definiendum does not occur
in the sample sentence.
"""

import pandas as pd
import numpy as np
import regex as re

file = "C:/Users/brand/Documents/Projects/XLdefgen/data/codwoe_test_de.csv"
data = pd.read_csv(file)

# Keep only exact matches (case-insensitive)
data['keep1'] = data.apply(lambda x: True if str(x['Wort']).lower() in x['Beispiel'].lower() else False, axis=1)

# Keep fuzzy matches (<= 2 character differences)
data['keep2'] = data.apply(
    lambda x: True if re.search("(?b)(" + str(x['Wort']).lower()
                                + "){e<=1}", str(x['Beispiel']).lower())
    else False, axis=1)

# Should change non-matches (in above code) to np.nan rather
# than False in order to easily delete them with this line.
data_clean = data.dropna()

# Print filtered data
print(data_clean[['Wort','Beispiel','keep2']])
print(data_clean.columns)

# Show fuzzy match examples
diffs = np.where(data['keep1'] != data['keep2'])
print(len(diffs[0]))
for diff in diffs[0][20:30]:
    print(data['Wort'].iloc[diff])
    print(data['Beispiel'].iloc[diff])
    print()