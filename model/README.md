# XLdefgen

## Train model

Acknowledgement: This script is adapted from Huggingface example code (https://github.com/huggingface/transformers/tree/master/examples/pytorch/translation).

The following are the arguments that may be passed when executing this script:\n
--dataset_name\n
--predict_with_generate\n
--dataset_config_name
--train_file
--num_beams
--max_source_length
--max_target_length
--val_max_target_length
--pad_to_max_length
--validation_file
--ignore_pad_token_for_loss****
--source_lang
--target_lang
--source_prefix
--preprocessing_num_workers
--overwrite_cache
--max_length

--model_name_or_path
--config_name
--tokenizer_name
--use_slow_tokenizer
--per_device_train_batch_size
--per_device_eval_batch_size
--learning_rate
--weight_decay
--num_train_epochs
--max_train_steps
--gradient_accumulation_steps
--lr_scheduler_type
--num_warmup_steps
--output_dir
--seed
--model_type
--push_to_hub
--hub_model_id
--hub_token
