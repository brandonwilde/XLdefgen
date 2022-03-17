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

def matcher(series):
    '''Mark definiendum for exact matches.
    Will still include affixes if present.'''

    sent = str(series['Beispiel'])
    word = str(series['Wort'])
    
    # Get start index of match
    match = re.search(word.lower(), sent.lower())
    if match:
        
        b_match = match.span()[0]+1
        e_match = match.span()[1]-1
        
        try:    # Get start index of word (may need to account for prefix)
            start = b_match - re.search('\W', sent[:b_match][::-1]).span()[0]
        except:
            start = 0
        
        try:    # Get end index of match (may need to account for affixes)
            end = e_match + re.search('\W', sent[e_match:]).span()[0]
        except:
            end = len(sent)
            
        return sent[:start]+'*'+sent[start:end]+'*'+sent[end:]
    
    else: # No match
        return np.nan


def matcher_1(series):
    """Mark definiendum for fuzzy matches.
    Allows up to one 'error' per match."""
    
    sent = str(series['Beispiel'])
    word = str(series['Wort'])
    
    if len(word) > 4:
        match = re.search("(?b)(" + word.lower() + "){e<=1}", sent.lower())

        if match:
            
            b_match = match.span()[0]+1
            e_match = match.span()[1]-1
            
            try:    # Get start index of word (may need to account for prefix)
                start = b_match - re.search('\W', sent[:b_match][::-1]).span()[0]
            except:
                start = 0
            
            try:    # Get end index of match (may need to account for affixes)
                end = e_match + re.search('\W', sent[e_match:]).span()[0]
            except:
                end = len(sent)
                
            return sent[:start]+'*'+sent[start:end]+'*'+sent[end:]
        
        else: # No match
            return np.nan
    
    else: # short word needs exact match
        match = re.search(word.lower(), sent.lower())
        if match:
            
            b_match = match.span()[0]+1
            e_match = match.span()[1]-1
            
            try:    # Get start index of word (may need to account for prefix)
                start = b_match - re.search('\W', sent[:b_match][::-1]).span()[0]
            except:
                start = 0
            
            try:    # Get end index of match (may need to account for affixes)
                end = e_match + re.search('\W', sent[e_match:]).span()[0]
            except:
                end = len(sent)
                
            return sent[:start]+'*'+sent[start:end]+'*'+sent[end:]
        
        else: # No match
            return np.nan
        
def matcher_2(series): # This seems to let in too many false positive matches (about half of the 67 added)
    """Mark definiendum for fuzzy matches.
    Allows up to one 'error' per match."""
    
    sent = str(series['Beispiel'])
    word = str(series['Wort'])
    
    if len(word) > 4:
        match = re.search("(?b)(" + word.lower() + "){e<=2}", sent.lower())

        if match:
            
            b_match = match.span()[0]+1
            e_match = match.span()[1]-1
            
            try:    # Get start index of word (may need to account for prefix)
                start = b_match - re.search('\W', sent[:b_match][::-1]).span()[0]
            except:
                start = 0
            
            try:    # Get end index of match (may need to account for affixes)
                end = e_match + re.search('\W', sent[e_match:]).span()[0]
            except:
                end = len(sent)
                
            return sent[:start]+'*'+sent[start:end]+'*'+sent[end:]
        
        else: # No match
            return False
    
    else: # short word needs exact match
        match = re.search(word.lower(), sent.lower())
        if match:
            
            b_match = match.span()[0]+1
            e_match = match.span()[1]-1
            
            try:    # Get start index of word (may need to account for prefix)
                start = b_match - re.search('\W', sent[:b_match][::-1]).span()[0]
            except:
                start = 0
            
            try:    # Get end index of match (may need to account for affixes)
                end = e_match + re.search('\W', sent[e_match:]).span()[0]
            except:
                end = len(sent)
                
            return sent[:start]+'*'+sent[start:end]+'*'+sent[end:]
        
        else: # No match
            return False

        
# data['keep'] = data.apply(matcher, axis=1)
data['keep1'] = data.apply(matcher_1, axis=1)
# data['keep2'] = data.apply(matcher_2, axis=1)
print(data)

# Should change non-matches (in above code) to np.nan rather
# than False in order to easily delete them with this line.
data_clean = data.dropna()
print(data_clean)

# Show fuzzy match examples
# diffs = np.where(data['keep'] != data['keep1'])
# print(len(diffs[0]))
# for diff in diffs[0][10:40]:
#     print(data['word'].iloc[diff])
#     print(data['pos'].iloc[diff])
#     print(data['Wort'].iloc[diff])
#     print(data['keep'].iloc[diff])
#     print(data['keep1'].iloc[diff])
#     # print(data['keep2'].iloc[diff])
#     print()

# Save marked data back to CSV file
data_clean.to_csv('codwoe_test_de_marked.csv')