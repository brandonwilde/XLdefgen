## Train model

Acknowledgement: This script is adapted from Huggingface example code (https://github.com/huggingface/transformers/tree/master/examples/pytorch/translation).

Example usage:

```bash
python run_translation_no_trainer.py \
	--file train_args_wmt.txt \
	--seed 42 \
	--output_dir /save/path
```

Below are all arguments that may be passed when executing this script.  
Arguments may also be passed as a text file to --file. Any arguments specified afterwards  
will override those indicated in the text file.

--model_name_or_path  
--config_name  
--model_type  

--dataset_name  
--dataset_config_name  
--train_file  		# CSV and JSON are your options  
--validation_file  	# CSV and JSON are your options  

--source_lang  
--target_lang  
--source_prefix  
--tokenizer_name  
--use_slow_tokenizer  
--max_source_length  
--max_target_length  
--pad_to_max_length  	# For TPU training (it disables dynamic padding)  
--val_max_target_length  
--max_length  - necessary?  
--preprocessing_num_workers  
--overwrite_cache  

--seed  
--per_device_train_batch_size  
--per_device_eval_batch_size  
--gradient_accumulation_steps  - could be good to experiment with  
--log_frequency
--num_train_epochs  
--max_train_steps  
--learning_rate  
--lr_scheduler_type  
--num_warmup_steps  
--weight_decay  
--ignore_pad_token_for_loss  
--predict_with_generate  
--num_beams  

--output_dir  
--push_to_hub  
--hub_model_id  
--hub_token  

Can use HF 'stas/mt5-tiny-random' model for testing.
