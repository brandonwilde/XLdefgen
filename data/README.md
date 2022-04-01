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
	--output_file codwoe_test_de_marked.json
```

Specify which language should be processed and how lenient the word-
matcher should be (kwarg: allow). The allowance is based on Minimum
Edit Distance, and will accept 0, 1, and 2 as inputs, with 0 restricting
the matcher to exact matches, and 2 permitting an MED of up to 2.
