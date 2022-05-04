## Prepare data

For use when preparing a dataset from a list of words, glosses, and
sample sentences. Use prepare_data.py to mark the definiendum in example sentences,
prepend the definiendum to example sentences, or both.
This will also remove any examples where the definiendum does not
occur in the example sentence.

Usage:
```bash
python prepare_data.py \
	--input_file codwoe/codwoe_test_de.csv \
	--source_lang de \
	--target_lang en \
	--mark \
	--demarcator "<extra_id_99>" \
	--allow 1 \
	--drop_columns \
	--output_file codwoe_test_de_marked.json
```

Specify which language should be processed and how lenient the wordmatcher
should be (kwarg: allow). The allowance is based on Minimum Edit Distance,
and will accept 0, 1, and 2 as inputs, with 0 restricting the matcher to
exact matches, and 2 permitting an MED of up to 2.

The outputs will be labeled 'input' and 'target'.

Below are all arguments that may be passed when executing this script.

--input_file (str)	# csv data file  
--source_lang (str)	# language for inputs  
--target_lang (str)	# language for targets (glosses)  
--mark (flag)		# if passed, definiendum will be marked in example sents  
--demarcator (str)	# symbol to use for marking definiendum  
--prepend (flag)		# if passed, definiendum will be prepended to example sentence  
--allow (int)		# maximum allowable MED when matching definiendum in example sent  
--drop_columns (flag)	# if passed, only input and target columns will be output  
--output_file (str)	# json data file  


## Split data

For use when you need to split training and test data, or if you need
to reduce the size of the training/validation datasets. Also works
when calling a dataset directly from the Hugging Face library.
Will save train and validation datasets as local JSON files.

Recommended usage:
```bash
python split_data.py --file data_args_wmt.txt \
```

Below is an example using all arguments that may be passed when
executing this script. Arguments may also be passed as a text file
to --file.

```bash
python split_data.py \
	--dataset_name_or_path wmt16 \
	--dataset_config_name de-en \
	--train_size 10000 \
	--validation_size 100 \
	--save_path wmt16_de-en \
	--seed 42
```
