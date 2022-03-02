# XLdefgen

## Assemble data

For use when you only need a small portion of an available dataset.
Use assemble_data.py to access a dataset from the Huggingface library,
reduce it down to your desired training set and validation set sizes,
and save local JSON files for each.

Example usage:

```bash
python assemble_data.py \
	--dataset_name wmt16 \
	--dataset_config_name de-en \
	--train_size 10000 \
	--validation_size 100 \
	--save_path wmt16_de-en \
	--seed 42
```

Below are all arguments that may be passed when executing this script.  
Arguments may also be passed as a text file to --file.

--dataset_name  
--dataset_config_name  
--train_size  
--save_path  
--seed  