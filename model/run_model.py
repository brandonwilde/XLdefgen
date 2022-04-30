#!/usr/bin/env python
# coding=utf-8
"""
Fine-tuning a 🤗 Transformers model on definition generation.
"""
import pdb
import argparse
import shlex
import wandb
import logging
import sys
import math
import os
import random
from pathlib import Path

import datasets
import numpy as np
import torch
from datasets import load_dataset, load_metric
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from huggingface_hub import Repository
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    MBartTokenizer,
    MBartTokenizerFast,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)
from transformers.file_utils import get_full_repo_name
from transformers.models.t5 import modeling_t5
from transformers.utils.versions import require_version

from custom_classes_and_fxns import (
    TokenizerWithXMask,
    MT5WithXMask,
    prepare_for_xattn,
    revise_residuals,
    get_batched_definienda
    # T5LayerSelfAttention
    )


logger = logging.getLogger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/translation/requirements.txt")

# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

# Allow arguments to be passed as a text file
class LoadFromFile(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        with values as f:
            # parse arguments in the file and store them in the target namespace
            line = f.read()
            parser.parse_args(shlex.split(line), namespace)

def str2bool(v):
    '''Accept string boolean values in arguments.'''
    if isinstance(v, bool):
        return v
    if v.lower() in ['true','t','yes','y','1']:
        return True
    if v.lower() in ['false','f','no','n','0']:
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

# Parsing input arguments
def parse_args():

    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a text classification task",
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
    
    # Not currently implemented, but should be
    parser.add_argument(
        "--predict_with_generate",
        type=str2bool,
        default=True,
        help="",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", 
        type=str,
        default=None,
        help="A csv or a json file containing the training data."
    )

    parser.add_argument(
        "--num_beams",
        type=int,
        default=None,
        help="Number of beams to use for evaluation. This argument will be "
        "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``.",
    )

    parser.add_argument(
        "--max_source_length",
        type=int,
        default=128,
        help="The maximum total input sequence length after "
        "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=128,
        help="The maximum total sequence length for target text after "
        "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
        "during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--val_max_target_length",
        type=int,
        default=None,
        help="The maximum total sequence length for validation "
        "target text after tokenization.Sequences longer than this will be truncated, sequences shorter will be "
        "padded. Will default to `max_target_length`.This argument is also used to override the ``max_length`` "
        "param of ``model.generate``, which is used during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--pad_to_max_length",
        type=str2bool,
        default=False,
        help="Whether to pad all samples to model maximum sentence "
        "length. If False, will pad the samples dynamically when batching to the maximum length in the batch. More"
        "efficient on GPU but very bad for TPU.",
    )
    parser.add_argument(
        "--validation_file",
        type=str,
        default=None,
        help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--ignore_pad_token_for_loss",
        type=str2bool,
        default=True,
        help="Whether to ignore the tokens corresponding to " "padded labels in the loss computation or not.",
    )
    parser.add_argument(
        "--source_lang",
        type=str,
        default=None,
        help="Source language id."
    )
    parser.add_argument(
        "--target_lang",
        type=str,
        default=None,
        help="Target language id."
    )
    parser.add_argument(
        "--source_prefix",
        type=str,
        default=None,
        help="A prefix to add before every source text " "(useful for T5 models).",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache",
        type=str2bool,
        default=True,
        help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="google/mt5-small",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        # required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=2,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=2,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--log_frequency",
        type=int,
        default=None,
        help="Number of gradient accumulation steps prior to evaluating and logging model.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=10,
        help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="fine-tuned_model",
        help=("Where to store the final model. If output_dir = 'wandb_run' and report_to = 'wandb',"
              "then model will be stored under the run name generated by WandB."
        )
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="A seed for reproducible training."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--push_to_hub",
        default=None,
        action="store_true", 
        help="Whether or not to push the model to the Hub."
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument(
        "--hub_token",
        type=str, 
        help="The token to use to push to the Model Hub."
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default=None,
        help="Where to track metrics during training."
    )
    parser.add_argument(
        "--wandb_proj",
        type=str,
        default=None,
        help="The WandB project name for the current run."
    )
    parser.add_argument(
        "--notes",
        type=str,
        default=None,
        help="WandB notes to be associated with the current run."
    )
    parser.add_argument(
        "--tags",
        type=str,
        default=None,
        help="Comma-separated WandB tags to be associated with the current run."
    )
    parser.add_argument(
        "--data_task",
        type=str,
        default="definition",
        help="Specify what kind of data is being used. Should be translation or definition."
    )
    parser.add_argument(
        "--mask_context",
        type=str2bool,
        default=False,
        help="Whether or not to use a cross-attention mask during decoding."
    )
    parser.add_argument(
        "--demarcator",
        type=str,
        default="",
        help="The string/symbol used to demarcate the definiendum in the example sentence."
    )
    parser.add_argument(
        "--input_column",
        type=str,
        default="input",
        help="The data column header (minus language) to be used as model input."
    )
    parser.add_argument(
        "--resid_wt",
        type=float,
        default=0.5,
        help="Weight of residual connection in self-attention (0 is no residual, 0.5 is normal, 1 is no attention)."
    )
    parser.add_argument(
        "--min_length",
        type=int,
        default=0,
        help="Minimum length of generated output."
    )
    parser.add_argument(
        "--mask_eos",
        type=str2bool,
        default=False,
        help="If mask_context=True, then this will also mask the eos_token during cross-attention."
        )
    parser.add_argument(
        "--filter_definiendum", # Not implemented yet
        type=str2bool,
        default=False,
        help="Whether to disallow the definiendum from appearing in the definition."
        )
    parser.add_argument(
        "--no_repeat_ngram_size",
        type=int,
        default=0,
        help="Disallows ngrams of this size or greater being repeated in generated output."
        )
    parser.add_argument(
        "--early_stopping",
        type=str2bool,
        default=False,
        help="Stop once num_beams have been completed?"
        )
    parser.add_argument(
        "--truncate",
        type=str2bool,
        default=True,
        help="Whether to truncate inputs and targets to specified max levels."
        )
    parser.add_argument(
        "--ban_definienda",
        type=str2bool,
        default=False,
        help="Whether to disallow definienda from being generated in output."
        )
    
    args = parser.parse_args()

    # Sanity checks

    if args.dataset_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")

    if args.train_file is not None:
        extension = args.train_file.split(".")[-1]
        assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        
    if args.validation_file is not None:
        extension = args.validation_file.split(".")[-1]
        assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    if args.data_task is not None:
        assert args.data_task in ["translation", "definition"], "`data_task` should be translation or definition."
        
    return args



def main():
    # Parse the arguments
    args = parse_args()  
    
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        stream=sys.stdout,
        filemode='w'
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    ckpt_path = "./checkpoints/"
    if accelerator.is_main_process:
        if args.output_dir == "wandb_run" and args.report_to != "wandb":
            args.output_dir == ckpt_path + "fine-tuned_model"
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(ckpt_path + args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)
        elif args.output_dir is not None and args.output_dir != "wandb_run":
            os.makedirs(ckpt_path + args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    else:
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = args.train_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.


    # Load pretrained model and tokenizer
        
    # Revise T5 source code prior to instantiating any of its classes
    if args.resid_wt:
        revise_residuals(args.resid_wt)
    
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.model_name_or_path) # may edit this line
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
    
    # Update config with args
    config = config.from_dict(vars(args))
    
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        tokenizer = TokenizerWithXMask.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
        
    if args.model_name_or_path:
        model = MT5WithXMask.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    else:
        logger.info("Training new model from scratch")
        model = MT5WithXMask.from_config(config)
    
    # These special tokens are normally used for specific training objectives,
    # but we're not using them that way here. These are being used as TEMPORARY
    # markers to demarcate the where the definiendum is.
    # Make sure to mark definienda with these tokens prior to running the model.
 
    # special_tokens_dict = {"mask_token": "<MASK>", "sep_token": " <MASK>"}
    # tokenizer.add_special_tokens(special_tokens_dict)
    # model.resize_token_embeddings(len(tokenizer))
    
    for item in sorted(model.config.to_dict().items()):
        print(item)

    # Set decoder_start_token_id to the the language code of the target language (!)
    if model.config.decoder_start_token_id is None and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
        assert (
            args.target_lang is not None and args.source_lang is not None
        ), "mBart requires --target_lang and --source_lang"
        if isinstance(tokenizer, MBartTokenizer):
            model.config.decoder_start_token_id = tokenizer.lang_code_to_id[args.target_lang]
        else:
            model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(args.target_lang)

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    prefix = args.source_prefix if args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names

    # For translation we set the codes of our source and target languages (only useful for mBART, the others will
    # ignore those attributes).
    if isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
        if args.source_lang is not None:
            tokenizer.src_lang = args.source_lang
        if args.target_lang is not None:
            tokenizer.tgt_lang = args.target_lang

    # Get the language codes for input/target.
    source_lang = args.source_lang.split("_")[0]
    target_lang = args.target_lang.split("_")[0]

    # Temporarily set max_target_length for training.
    max_target_length = args.max_target_length
    padding = "max_length" if args.pad_to_max_length else False
        
    # May need to edit this if train and validation datasets have different data_task
    def preprocess_function(examples):
        
        
        # Set input and target keys
        if args.data_task == "definition":
            if args.input_column == "input":
                input_label = args.input_column
            else:
                input_label = target_lang + '_' + args.input_column
            target_label = "target"
            # target_lang + "_gloss"
        
        elif args.data_task == "translation":
            input_label = source_lang
            target_label = target_lang
        
        else:
            raise KeyError("Need to specify how to handle data if data_task"
                           "is not 'definition' or 'translation'.")
            
        inputs = [ex[input_label] for ex in examples[args.data_task]]
        targets = [ex[target_label] for ex in examples[args.data_task]]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=padding, truncation=args.truncate)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=args.truncate)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    with accelerator.main_process_first():             
        processed_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not args.overwrite_cache,
        # load_from_cache_file=False,
        desc="Running tokenizer on dataset",
        )
        
        for ex in processed_datasets["train"]["input_ids"][:5]:
            print("Example input length:", len(ex))
        
        # This doesn't work well with mixed inputs
        # Filter out inputs where the definiendum has been cut off in truncation.
        if args.demarcator is not None:
            init_length = len(processed_datasets["train"])
            
            # check for both demarcators
            demarc_id = tokenizer.convert_tokens_to_ids(args.demarcator)
            processed_datasets = processed_datasets.filter(lambda ex:
                                                            ex['input_ids'].count(demarc_id) == 2
                                                            )
            end_length = len(processed_datasets["train"])
            dropped = init_length - end_length
            print(f"{dropped} examples filtered out due to definiendum getting lost in truncation.")
            
        
        # Add cross-attention mask, remove definiendum span markers
        if args.mask_context:
            processed_datasets = processed_datasets.map(lambda x:
                                                    prepare_for_xattn(x, tokenizer, args.demarcator, args.mask_eos),
                                                    desc="Adding cross-attention mask")
        
    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]
    print("Num eval examples: ", eval_dataset)
    
    # Confirm attention mask works properly
    if args.mask_context:
        examp = eval_dataset[-2]
        examp_full = zip(tokenizer.convert_ids_to_tokens(examp['input_ids']),
                    examp['input_ids'],
                    examp['attention_mask'],
                    examp['cross_attention_mask']
                    )
        for tok in examp_full:
            print("Eval example:", tok)
        print()

     # DataLoaders creation:
    label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if accelerator.use_fp16 else None,
        )

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Get a list of the definienda for each batch (to ban them in output)
    demarc_id = tokenizer.convert_tokens_to_ids(args.demarcator)
    if args.ban_definienda:
        definienda = get_batched_definienda(eval_dataloader, args.mask_context, demarc_id, tokenizer)
    
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # If no log_frequency set, default to epoch logging
    if args.log_frequency is None:
        args.log_frequency = num_update_steps_per_epoch
        
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    
    metric = load_metric("sacrebleu")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels
    

    # Train!
       
    if args.report_to == "wandb":

        tags = args.tags.split(",") if args.tags else None

        # Report hyperparameters
        wb_config = model.config.to_dict()
        # wb_config.update(vars(args))
                
        wandb.init(
            project=args.wandb_proj,
            notes=args.notes,
            tags=tags,
            config=wb_config,
            allow_val_change=True
            )
        
        if args.output_dir == "wandb_run":
            args.output_dir = wandb.run.name
    
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
      
    for epoch in range(args.num_train_epochs):
        for train_step, batch in enumerate(train_dataloader):
            model.train()
            outputs = model(**batch)
            loss = outputs.loss             # Gradient accumulating
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if (train_step+1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()       # Reset gradients
                progress_bar.update(1)      # Actually counts grad_accum steps rather than batches
                completed_steps += 1        # Actually counts grad_accum steps rather than batches
                if args.report_to == "wandb":
                    wandb.log({'epoch': epoch+(train_step+1)/len(train_dataloader),
                               'batch': epoch*len(train_dataloader)+train_step+1,
                               'effective_batch': completed_steps,
                               'train/loss': loss,
                               })

                if completed_steps % args.log_frequency == 0 or train_step == len(train_dataloader) - 1:    # Evaluate by spec. frequency
                    model.eval()
                    loss = 0
                    
                    print("Num eval batches:", len(eval_dataloader))
            
                    if args.val_max_target_length is None:
                        args.val_max_target_length = args.max_target_length
            
                    gen_kwargs = {
                        "max_length": args.val_max_target_length if args is not None else config.max_length,
                        "num_beams": args.num_beams,
                        "no_repeat_ngram_size": model.config.no_repeat_ngram_size,
                        "repetition_penalty": model.config.repetition_penalty,
                        "early_stopping": model.config.early_stopping,
                    }
                    for eval_step, batch in enumerate(eval_dataloader):
                        with torch.no_grad():
                            
                            outputs = model(**batch)
                            loss += outputs.loss             # Gradient accumulating
                            
                            outputs = accelerator.unwrap_model(model).generate(
                                batch["input_ids"],
                                attention_mask=batch["attention_mask"],
                                bad_words_ids=definienda[eval_step],
                                **gen_kwargs,
                                return_dict_in_generate=True,
                                output_scores=True
                            )
                            
                            generated_tokens = outputs.sequences
                            
                            generated_tokens = accelerator.pad_across_processes(
                                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                            )
                            labels = batch["labels"]
                            if not args.pad_to_max_length:
                                # If we did not pad to max length, we need to pad the labels too
                                labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)
            
                            generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
                            labels = accelerator.gather(labels).cpu().numpy()
                                    
                            if args.ignore_pad_token_for_loss:
                                # Replace -100 in the labels as we can't decode them.
                                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            
                            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            
                            decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    
                            metric.add_batch(predictions=decoded_preds, references=decoded_labels)
                    print("Epochs completed:", epoch+(train_step+1)/len(train_dataloader))
                    for pred in zip(decoded_preds, decoded_labels):
                        print()
                        print("Pred: ", pred[0])
                        print("Actual: ", pred[1])
                    val_loss = loss/len(eval_dataloader)
                    val_ppl = round(math.exp(val_loss),4)
                    eval_metric = metric.compute()
                    logger.info({"bleu": eval_metric["score"]})
                    if args.report_to == "wandb":
                        wandb.log({'epoch': epoch+(train_step+1)/len(train_dataloader),
                                   'batch': epoch*len(train_dataloader)+train_step+1,
                                   'effective_batch': completed_steps,
                                   'eval/loss': val_loss,
                                   'eval/perplexity': val_ppl,
                                   'eval/bleu': eval_metric['score'],
                                   })
                if completed_steps >= args.max_train_steps:
                    break

        # Save at end of each epoch
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(ckpt_path + args.output_dir, save_function=accelerator.save)
        if accelerator.is_main_process:         # Only do once, if distributed
            tokenizer.save_pretrained(ckpt_path + args.output_dir)
            if args.push_to_hub:
                if epoch < args.num_train_epochs - 1:
                    repo.push_to_hub(
                        commit_message=f"Training in progress - epoch {epoch}", blocking=False, auto_lfs_prune=True
                    )
                else:
                    repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)

if __name__ == "__main__":

    main()
