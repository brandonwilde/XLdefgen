#!/usr/bin/env python
# coding=utf-8
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

# Keep fuzzy matches (<= 1 character differences)
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
    

def matcher(series):
    '''Mark definiendum for exact matches.
    Will still include affixes if present.'''

    sent = series['Beispiel']
    word = series['Wort']
    
    # Get start index of match
    start = sent.lower().find(str(word).lower())
    if start != -1:
        try:    # Get end index of match (may need to account for affixes)
            end = start + re.search('\W', sent[start:]).span()[0]
        except: # Definiendum is at end of string
            return sent[:start]+'*'+sent[start:]+'*'
        else:
            return sent[:start]+'*'+sent[start:end]+'*'+sent[end:]
    else: # No exact match
        return False


# def matcher_fuzzy(series):
    # """Mark definiendum for fuzzy matches.
    # Allows up to one 'error' per match."""
    
#     match = re.search("(?b)(" + str(series['Wort']).lower()
#                                 + "){e<=1}", str(series['Beispiel']).lower())
#     if match:
#         sent = series['Beispiel']
#         s = match.span()
#         return sent[:s[0]]+'*'+sent[s[0]:s[1]]+'*'+sent[s[1]:]
#     else:
#         return False
    
data['keep3'] = data.apply(matcher, axis=1)
print(data)