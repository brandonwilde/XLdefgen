#!/usr/bin/env python
# coding=utf-8
"""
Mark the definiendum in each sample sentence, and remove any examples
where the definiendum does not occur in the sample sentence.
All prefixes and suffixes will be included so long as the root definiendum
is matched.
"""

import argparse
import pandas as pd
import numpy as np
import regex as re


# Parsing input arguments
def parse_args():

    parser = argparse.ArgumentParser(
        description="Clean and mark the data",
    )
        
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help=("The csv file holding the raw data. Columns should be labeled with language code "
              "and the type of data (e.g. en_gloss). Columns must include 'en_word', 'en_gloss', "
              "and 'en_example' or the language's equivalent of these.")
    )
    
    parser.add_argument(
        "--source_lang",
        type=str,
        default="en",
        help=("The language of the input data. Should match the language code in the input "
              "file's column headers."),
    )
    
    parser.add_argument(
        "--target_lang",
        type=str,
        default="en",
        help=("The language of the target data. Should match the language code in the input "
              "file's 'gloss' header."),
    )
    
    parser.add_argument(
        "--allow",
        type=int,
        default=0,
        choices=[0,1,2],
        help=("The allowable minimum edit distance when matching the word to its occurence within "
        "the sentence. May be 0, 1, or 2.")
    )
    
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="The language to create marked sentences in.",
    )

    parser.add_argument(
        "--demarcator",
        type=str,
        default="*",
        help="The string or symbol that should be used to demarcate the definiendum.",
    )

    parser.add_argument(
        "--mark",
        action="store_true",
        help="If passed, will demarcate definiendum within example sentence.",
    )
    
    parser.add_argument(
        "--prepend",
        action="store_true",
        help="If passed, will prepend definiendum to example/marked sentence.",
    )
    parser.add_argument(
        "--drop_columns",
        action="store_true",
        help="If passed, will drop all unnecessary columns after preparing data.",
    )

    args = parser.parse_args()
    
    # assert args.lang in ["en", "de"], "language must be 'en' or 'de'."

    return args

# Input args to specify train/val/test split - which should be included
# and how much data to put in each split

def matcher_0(series, lang="en", demarcator="*"):
    '''Mark definiendum for exact matches.
    Will still include affixes if present.'''
    
    sent = str(series[lang+'_example'])
    word = str(series[lang+'_word'])
    
    # print(word)
    # print(sent)
    
    # Get start index of match
    match = re.search(word.lower(), sent.lower())
    if match:
        
        b_match = match.span()[0]+1
        e_match = match.span()[1]-1
        
        try:    # Get start index of word (may need to account for prefix)
            start = b_match - re.search('\W', sent[:b_match][::-1]).span()[0]
        except:
            start = 0
        
        try:    # Get end index of match (may need to account for suffixes)
            end = e_match + re.search('\W', sent[e_match:]).span()[0]
        except:
            end = len(sent)
            
        return sent[:start]+' '+demarcator+' '+sent[start:end]+' '+demarcator+' '+sent[end:]
    
    else: # No match
        return np.nan


def matcher_1(series, lang="en", demarcator="*"):
    """Mark definiendum for fuzzy matches.
    Allows up to one 'error' per match."""
           
    sent = str(series[lang+'_example'])
    word = str(series[lang+'_word'])
    
    if len(word) > 4:
        match = re.search("(?b)(" + word.lower() + "){e<=1}", sent.lower())

        if match:
            
            b_match = match.span()[0]+1
            e_match = match.span()[1]-1
            
            try:    # Get start index of word (may need to account for prefix)
                start = b_match - re.search('\W', sent[:b_match][::-1]).span()[0]
            except:
                start = 0
            
            try:    # Get end index of match (may need to account for suffixes)
                end = e_match + re.search('\W', sent[e_match:]).span()[0]
            except:
                end = len(sent)
                
            return sent[:start]+demarcator+' '+sent[start:end]+' '+demarcator+sent[end:]
        
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
            
            try:    # Get end index of match (may need to account for suffixes)
                end = e_match + re.search('\W', sent[e_match:]).span()[0]
            except:
                end = len(sent)
                
            return sent[:start]+demarcator+' '+sent[start:end]+' '+demarcator+sent[end:]
        
        else: # No match
            return np.nan
        
def matcher_2(series, lang="en", demarcator="*"): # This seems to let in too many false positive matches (about half of the 67 added)
    """Mark definiendum for fuzzy matches.
    Allows up to two 'errors' per match."""
            
    sent = str(series[lang+'_example'])
    word = str(series[lang+'_word'])
    
    if len(word) > 4:
        match = re.search("(?b)(" + word.lower() + "){e<=2}", sent.lower())

        if match:
            
            b_match = match.span()[0]+1
            e_match = match.span()[1]-1
            
            try:    # Get start index of word (may need to account for prefix)
                start = b_match - re.search('\W', sent[:b_match][::-1]).span()[0]
            except:
                start = 0
            
            try:    # Get end index of match (may need to account for suffixes)
                end = e_match + re.search('\W', sent[e_match:]).span()[0]
            except:
                end = len(sent)
                
            return sent[:start]+demarcator+' '+sent[start:end]+' '+demarcator+sent[end:]
        
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
            
            try:    # Get end index of match (may need to account for suffixes)
                end = e_match + re.search('\W', sent[e_match:]).span()[0]
            except:
                end = len(sent)
                
            return sent[:start]+demarcator+' '+sent[start:end]+' '+demarcator+sent[end:]
        
        else: # No match
            return np.nan


def main():
    # Parse the arguments
    args = parse_args()
    
    if args.input_file is None: # Or if not running from command line
        in_file = "C:/Users/brand/Documents/Projects/XLdefgen/data/codwoe_train_en.csv"
    else:
        in_file = args.input_file
        
    data = pd.read_csv(in_file)
    start_length = len(data)
    
    print(data)
    
    # Set default column for concatenation
    concat_col = args.source_lang + "_example"
    
    if args.mark:     
        if args.allow == 0:
            data[args.source_lang+'_marked'] = data.apply(matcher_0, lang=args.source_lang, demarcator=args.demarcator, axis=1)
        elif args.allow == 1:
            data[args.source_lang+'_marked'] = data.apply(matcher_1, lang=args.source_lang, demarcator=args.demarcator, axis=1)
        elif args.allow == 2:
            data[args.source_lang+'_marked'] = data.apply(matcher_2, lang=args.source_lang, demarcator=args.demarcator, axis=1)
        
        data['input'] = data[args.source_lang+'_marked']
        concat_col = args.source_lang + "_marked"
    
    if args.prepend:
        data[args.source_lang+'_prepend'] = data.apply(lambda x: str(x[args.source_lang+'_word']) + '. ' + str(x[concat_col]), axis=1)
        data['input'] = data[args.source_lang+'_prepend']
    
    data['target'] = data[args.target_lang+'_gloss']
    
    data_clean = data.dropna()  # Remove data based on nan's in marked column
    end_length = len(data_clean)
    
    print(data_clean)
    print(start_length-end_length, "samples removed due to the target word not being found in the example sentence.")
    
    # Remove unnecessary columns
    if args.drop_columns:
        keep_columns = ["input", "target"]
        data_clean = data_clean.drop(columns=[col for col in data_clean.columns.values if col not in keep_columns])
    
    # Show fuzzy match examples
    # Should change non-matches (in 'matcher' functions) to False rather
    # than np.nan in order to easily compare results.
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
    
    # Save marked data to JSON file      
    
    # data_clean.to_json("temp_file.json", orient='records', lines=True)
    # temp_file = data_clean.to_json(orient='records', lines=True)
    
    import json
    # l = []
    data_entries = data_clean.to_dict(orient = 'records')
    with open(args.output_file, 'w') as f:
        for line in data_entries:
            d = {}
            d["definition"] = line
            f.write(json.dumps(d, sort_keys=False) + '\n') 
    #         l.append(d)
    # json.dumps(l)
    
    # Add additional dict layer specifying task
    # with open("temp_file.json", 'r') as f:
    #     file_lines = [''.join(['{"definition":', line.strip(), "}", '\n']) for line in f.readlines()]
    
    # with open(args.output_file, 'w') as f:
    #     f.writelines(file_lines) 
    
    print("Formatted data saved in", args.output_file)

if __name__ == "__main__":

    main()