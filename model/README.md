## Train model

Acknowledgement: This script is adapted from Huggingface example code (https://github.com/huggingface/transformers/tree/master/examples/pytorch/translation).

Example usage:

```bash
python run_model.py --file train_args_codwoe.txt
```

OR (for testing on small GPU)

```bash
python run_model.py --file train_args_wmt_tiny.txt
```

Below are all arguments that may be passed when executing this script.  
Arguments may also be passed as a text file to --file. Any arguments
specified afterwards will override those indicated in the text file.

If train data and validation data are not of the same type, then 'input'
should be made the column header of the desired data prior to running
this code, and 'input' should be passed to input_column.  
This can handle applying an attention mask selectively, but not if
definiendum span markers are also used in the data not to be masked.  

--model_name_or_path  
--config_name  (str)	# if not default  
--model_type  

--dataset_task (str)	# defintion or translation - should match data field  
--dataset_name  (str)	# if loading from Hugging Face hub  
--dataset_config_name  	# if not default  
--train_file  		# CSV and JSON are your options  
--validation_file  	# CSV and JSON are your options  

--source_lang  (str)	# two-letter code - should match data labels  
--target_lang  (str)	# two-letter code - should match data labels  
--source_prefix  (str)	# text to prepend to each input  
--tokenizer_name  (str)	# if not model default  
--use_slow_tokenizer  (flag)  
--max_source_length  
--max_target_length  
--pad_to_max_length  	# For TPU training (it disables dynamic padding)  
--val_max_target_length  
--max_length  - may not be necessary?  
--preprocessing_num_workers  
--overwrite_cache  
--input_column  (str)	# column header ('input', 'marked', 'prepend', etc.)  
--demarcator  (str)	# definiendum marker ("*", "<extra_id_99>", etc.)  
--mask_context  (bool)	# whether to mask context based on definiendum markers  
--resid_wt (float)	# 0-1 (0 = no residual, 0.5 = normal, 1 = no attention)

--seed  
--per_device_train_batch_size  
--per_device_eval_batch_size  
--gradient_accumulation_steps  (int)	# num steps to combine in effective batch  
--log_frequency  (int)	# num effective batches between each evaluation  
--num_train_epochs  
--max_train_steps  
--learning_rate  
--lr_scheduler_type  (str)	# linear, cosine, constant, etc.  
--num_warmup_steps  
--weight_decay  
--ignore_pad_token_for_loss  
--predict_with_generate  	# not built in yet  
--num_beams  

--output_dir  (str)	# name of folder or 'wandb_run' to defer to name of wandb run  
--report_to  (str)	# wandb or None  
--wandb_proj  (str)	# name of wandb project  
--notes  (str)		# notes to associate with current run  
--tags  (str)		# comma-separated tags to associate with current run  

--push_to_hub  
--hub_model_id  
--hub_token  

Can use HF 'stas/mt5-tiny-random' model for testing.