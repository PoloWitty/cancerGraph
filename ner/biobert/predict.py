#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for token classification.
"""
# This is modified from huggingface example
# author: Yangzhe Peng
# date: 2023/03/07

import logging
import os
import ipdb
import json
import sys
import tqdm
from dataclasses import dataclass, field
from typing import Optional

import datasets
import evaluate
import numpy as np
from datasets import ClassLabel, load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    HfArgumentParser,
    PretrainedConfig,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from modeling_multiNER import BERTMultiNER2
import torch
from multiprocessing import Pool
import math
import pickle

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.27.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/token-classification/requirements.txt")

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    use_multiNERHead: bool = field(
        default=True,
        metadata={"help": "Will use different Tokenclassification header for different entity type"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    predict_file: str = field(
        default=None,
        metadata={"help": "a csv or JSON file."},
    )
    text_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of text to input in the file (a csv or JSON file)."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. If set, sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    label_all_tokens: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to put the label for one word on all tokens of generated by that word or just on the "
                "one (in which case the other tokens will have a padding index)."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    decode_worker_num: Optional[int] = field(
        default=4,
        metadata={"help": 'use how many cpu workers to decode the output of model'}
    )

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        data_files = {}
        data_files["predict"] = data_args.predict_file
        extension = data_args.predict_file.split(".")[-1] 
        raw_datasets = load_dataset(extension if extension!='jsonl' else 'json', data_files=data_files, cache_dir=model_args.cache_dir)
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    column_names = raw_datasets["predict"].column_names

    if data_args.text_column_name is not None:
        text_column_name = data_args.text_column_name
    elif "tokens" in column_names:
        text_column_name = "tokens"
    else:
        text_column_name = column_names[0]

    # If the labels are of type ClassLabel, they are already integers and we have the map stored somewhere.
    # Otherwise, we have to get the list of labels manually.
    label_list = ['B','I','O']
    label_to_id = {l: i for i, l in enumerate(label_list)}

    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    tokenizer_name_or_path = model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path
    if config.model_type in {"bloom", "gpt2", "roberta"}:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=True,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            add_prefix_space=True,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=True,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    if model_args.use_multiNERHead :
        model = BERTMultiNER2.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        model = AutoModelForTokenClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        )
    model.config.entityType2id = {
        "predict": 0,
        "Disease": 1,
        "Drug/Chem": 2,
        "Gene/Protein": 3,
        "Species": 4
    }

    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models at"
            " https://huggingface.co/transformers/index.html#supported-frameworks to find the model types that meet"
            " this requirement"
        )

    # Model has labels -> use them.
    if model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id:
        if sorted(model.config.label2id.keys()) == sorted(label_list):
            # Reorganize `label_list` to match the ordering of the model.
            label_list = [model.config.id2label[i] for i in range(num_labels)]
            label_to_id = {l: i for i, l in enumerate(label_list)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {sorted(model.config.label2id.keys())}, dataset labels:"
                f" {sorted(label_list)}.\nIgnoring the model labels as a result.",
            )

    # Set the correspondences label/ID inside the model config
    model.config.label2id = {l: i for i, l in enumerate(label_list)}
    model.config.id2label = {i: l for i, l in enumerate(label_list)}

    # Preprocessing the dataset
    # Padding strategy
    padding = "max_length" if data_args.pad_to_max_length else False

    # Tokenize all texts and align the labels with them.
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            padding=padding,
            truncation=True,
            max_length=data_args.max_seq_length,
            is_split_into_words=True
        )
        word_masks = []
        for i, label in enumerate(examples[text_column_name]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            word_mask = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    word_mask.append(-1)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    word_mask.append(1)
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    if data_args.label_all_tokens:
                        word_mask.append(1)
                    else:
                        word_mask.append(0)
                previous_word_idx = word_idx

            word_masks.append(word_mask)

        entityTypeIds = [[model.config.entityType2id['predict']]*len(tokenized_input) for tokenized_input in tokenized_inputs['input_ids']]
        tokenized_inputs["entity_type_ids"] = entityTypeIds
        tokenized_inputs['word_mask'] = word_masks
        return tokenized_inputs

    if training_args.do_predict:
        if "predict" not in raw_datasets:
            raise ValueError("do predict requires a dataset")
        predict_dataset = raw_datasets["predict"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                tokenize_and_align_labels,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                desc="Running tokenizer on prediction dataset",
            )
    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    global decode_batch
    def decode_batch(args):
        worker_num,batch_size,possibilities,predictions,word_mask,predict_dataset = args

        idx2type = ['disease','chemical','gene','species']
        idx2label = ['B','I','O']

        true_predictions = []; true_inputs = []
        for i in tqdm.trange(0,batch_size,desc='worker %d'%worker_num,position=worker_num):
            # process output
            true_prediction = []
            for j in range(word_mask.shape[1]):
                if word_mask[i][j]==1:
                    labels = predictions[i][j] # label
                    possibility = possibilities[i][j]
                    if (labels==2).all(): # all predict O
                        true_prediction.append('O')
                    else:
                        possibility = np.where(labels!=2,possibility,0)
                        t = np.argmax(possibility)
                        l = labels[t]
                        true_prediction.append(idx2label[l]+'-'+idx2type[t])
            # get word level input
            true_input = []
            indices = np.nonzero(word_mask[i]==1)[0]
            for idx in range(1,len(indices)):
                prev_indice = indices[idx-1]
                indice = indices[idx]
                true_input.append(tokenizer.decode(predict_dataset[i]['input_ids'][prev_indice:indice]))
            # find the last one
            for idx in range(indice,word_mask.shape[1]):
                if word_mask[i][idx]==-1:
                    special_idx=idx
                    break
            true_input.append(tokenizer.decode(predict_dataset[i]['input_ids'][indice:special_idx]))
            true_inputs.append(true_input)
            true_predictions.append(true_prediction)
        return [true_inputs,true_predictions]
    
    # Predict
    if training_args.do_predict:
        logger.info("*** Predict ***")

        # (preds,word_mask), _ , metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")
        # pickle.dump((preds,word_mask),open('preds_wordmask.pkl','wb'))
        # del trainer
        # torch.cuda.empty_cache()
        preds,word_mask = pickle.load(open('preds_wordmask.pkl','rb'))
        possibilities = []
        predictions = []
        for pred in tqdm.tqdm(preds,desc='postprocess output'):
            possibility = pred.max(axis=-1)
            possibilities.append(np.expand_dims(possibility,axis=-1))
            
            prediction = np.argmax(pred,axis=-1)
            predictions.append(np.expand_dims(prediction,axis=-1))

        possibilities = np.concatenate(possibilities,axis=-1) # determin which type
        predictions = np.concatenate(predictions,axis=-1)

        true_predictions = []
        true_inputs = []

        total_num = word_mask.shape[0]
        batch_size = math.ceil(total_num/data_args.decode_worker_num)
        process_args = []
        i = 0
        while (i+1)*batch_size < total_num:
            s,e = i*batch_size,(i+1)*batch_size
            process_args.append((i,batch_size,possibilities[s:e],predictions[s:e],word_mask[s:e],predict_dataset.select(range(s,e))))
            i+=1
        process_args.append((i,total_num-i*batch_size,possibilities[i*batch_size:total_num],predictions[i*batch_size:total_num],word_mask[i*batch_size:total_num],predict_dataset.select(range(i*batch_size,total_num))))

        with Pool(data_args.decode_worker_num) as p:
            print('decoding')
            outputs = p.map(decode_batch, process_args)
        
        pickle.dump(outputs,open('output.pkl','wb'))
        
        true_inputs = []; true_predictions = []
        for i in range(data_args.decode_worker_num):
            true_inputs+=outputs[i][0]
            true_predictions+=outputs[i][1]

        # save output to dataset and remove model or tokenizer output
        predict_dataset = predict_dataset.map(lambda example,idx: {'ner_abstract':true_predictions[idx],'abstract':true_inputs[idx]},with_indices=True,remove_columns=['input_ids','token_type_ids','attention_mask','entity_type_ids','word_mask'])
        # trainer.log_metrics("predict", metrics)
        # trainer.save_metrics("predict", metrics)

        # Save predictions
        predict_dataset.to_json(training_args.output_dir+'predictions.jsonl')

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "token-classification"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()