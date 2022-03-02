# -*- coding: utf-8 -*-
"""
Download, reduce resize of dataset.
Saves data as json files.
"""

import argparse
import shlex
from datasets import load_dataset

# Allow arguments to be passed as a text file
class LoadFromFile(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        with values as f:
            # parse arguments in the file and store them in the target namespace
            line = f.read()
            parser.parse_args(shlex.split(line), namespace)


# Parsing input arguments
def parse_args():

    parser = argparse.ArgumentParser(
        description="Access a Huggingface dataset and save it as a json file.",
    )
        
    parser.add_argument(
        "--file",
        type=open,
        action=LoadFromFile
    )
    
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )

    parser.add_argument(
        "--train_size",
        type=int,
        default=None,
        help="Reduced size of the training dataset.",
    )
    
    parser.add_argument(
        "--validation_size",
        type=int,
        default=None,
        help="Reduced size of the validation dataset.",
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default="reduced_data",
        help="Where to store the reduced datasets."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="A seed for reproducible training."
    )

    args = parser.parse_args()

    return args

def main():
    # Parse the arguments
    args = parse_args()

    # dataset_name = 'wmt16'
    # dataset_config_name = 'de-en'
    # train_size = 10000
    # validation_size = 100
    # seed = 42
    # save_path = 'wmt16_de-en'
    save_filetype = 'json'      # Formatting messes up with csv
    
    # assert save_filetype in ["csv", "json"], "`train_file` should be a csv or a json file."
    
    if args.dataset_name is not None:
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    
    if args.train_size is not None:
        train_data = raw_datasets["train"].shuffle(seed=args.seed).select(range(args.train_size))
    
    if args.validation_size is not None:
        val_data = raw_datasets["validation"].shuffle(seed=args.seed).select(range(args.validation_size))
    
    if save_filetype == 'json':
        train_data.to_json(args.save_path + '_train.json')
        val_data.to_json(args.save_path + '_val.json')
    # if save_filetype == 'csv':
    #     train_data.to_csv(save_path + '_train.csv')
    #     val_data.to_csv(save_path + '_val.csv')
    
    
    print(f'Datasets saved to {args.save_path}_<data_split>.{save_filetype}')



if __name__ == "__main__":

    main()

#%%

# train_file = "wmt16_de-en_train.json"
# validation_file = "wmt16_de-en_val.json"

# if train_file is not None:
#     extension = train_file.split(".")[-1]
    
# data_files = {}
# if train_file is not None:
#     data_files["train"] = train_file
# if validation_file is not None:
#     data_files["validation"] = validation_file
# extension = train_file.split(".")[-1]
# loaded_datasets = load_dataset(extension, data_files=data_files)
    