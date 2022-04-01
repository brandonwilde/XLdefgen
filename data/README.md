## Assemble training data

For use when you only need a small portion of an available dataset.
Use assemble_data.py to access a dataset from the Huggingface library,
reduce it down to your desired training set and validation set sizes,
and save local JSON files for each.

Recommended usage:
```bash
python assemble_data.py --file data_args_wmt.txt \
```

Below is an example using all arguments that may be passed when
executing this script. Arguments may also be passed as a text file
to --file.

```bash
python assemble_data.py \
	--dataset_name wmt16 \
	--dataset_config_name de-en \
	--train_size 10000 \
	--validation_size 100 \
	--save_path wmt16_de-en \
	--seed 42
```

## Mark data

For use when preparing a dataset from a list of words, glosses, and  
sample sentences. Use mark_data.py to mark the definiendum in each  
sample sentence, and remove any examples where the definiendum does not  
occur in the sample sentence.

Usage:
```bash
python mark_data.py \
	--input_file codwoe_test_de.csv \
	--lang de \
	--allow 1 \
	--output_file codwoe_test_de_marked.csv
```

Specify which language should be processed and how lenient the word-
matcher should be (kwarg: allow). The allowance is based on Minimum
Edit Distance, and will accept 0, 1, and 2 as inputs, with 0 restricting
the matcher to exact matches, and 2 permitting an MED of up to 2.
