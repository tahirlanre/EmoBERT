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
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...) on a text file or a dataset.
Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=masked-lm
"""
# You can also adapt this script on your own masked language modeling task. Pointers for this are left as comments.

import logging
import math
import os
import pickle
import sys
from dataclasses import dataclass, field
from typing import Optional
import re

from datasets import load_dataset

import transformers
from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed
)
from transformers.trainer_utils import is_main_process
from data_collator import DataCollatorForLanguageModeling

logger = logging.getLogger(__name__)

emo_lexicon = pickle.load(open('data/emolex_emo_words.pkl', 'rb'))

# emoji regex for smiley & emotion emojis from emojitracker.com
all_emoji_regex = re.compile(r'['
                             '\U0000263a\U00002764\U0001f479-\U0001f47b\U0001f47d-'
                             '\U0001f480\U0001f48b-\U0001f48c\U0001f493-\U0001f49f\U0001f4a2-'
                             '\U0001f4a6\U0001f4a8-\U0001f4a9\U0001f4ab-\U0001f4ad\U0001f4af\U0001f600-'
                             '\U0001f640\U0001f648-\U0001f64a'
                             ']', re.UNICODE)
          

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    train_file: str = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    emo_mlm_probability: float = field(
        default=0.3, metadata={"help": "Ratio of emolex tokens to mask for masked language modelling loss"}
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )

    def __post_init__(self):
        if self.train_file is None and self.validation_file is None:
            raise ValueError("Need a training/validation file")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."

                        
def _process_tokens(tokens):
    for i in range(len(tokens)):
        if tokens[i].startswith('http'):
            tokens[i] = '<url>'
        elif tokens[i].startswith('@'):
            tokens[i] = '<user>'
    return " ".join(tokens)

def _emoji_present(text):
    return bool(re.search(all_emoji_regex.pattern, text))

def _emolex_present(text):
    return bool(set(emo_lexicon).intersection(text.lower().split()))

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

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f'Device: {training_args.device}, n_gpu: {training_args.n_gpu}'
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    data_files = {}
    if data_args.train_file is not None:
        data_files['train'] = data_args.train_file
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
    extension = data_args.train_file.split(".")[-1]
    if extension == 'txt':
        extension = 'text'
    datasets = load_dataset(extension, data_files=data_files, column_names=[i for i in range(16)])

    # Load pretrained model and tokenizer
    model_checkpoint = model_args.model_name_or_path if model_args.model_name_or_path else 'bert-base-uncased'
    config = BertConfig.from_pretrained(model_checkpoint)

    tokenizer = BertTokenizer.from_pretrained(model_checkpoint)

    model = BertForMaskedLM.from_pretrained(
        model_checkpoint,
        config=config,
    )

    model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we filter all the texts if they contain emojis.
    if training_args.do_train:
        column_names = datasets["train"].column_names
    else:
        column_names = datasets["validation"].column_names
    text_column_name = "text" if "text" in column_names else column_names[3]

    def filter_function(examples):
        return _emoji_present(examples[text_column_name]) # TODO and _emolex_present(examples['text'])

    filtered_datasets = datasets.filter(
        filter_function,
        num_proc=data_args.preprocessing_num_workers
    )

    # We tokenize all texts and change all occurrences of url and usernames to special tags
    padding = "max_length" if data_args.pad_to_max_length else False

    def tokenize_function(examples):
        # examples["text"] = [line for line in examples[text_column_name] if len(line) > 0 and not line.isspace()]
        examples["text"] = [_process_tokens(text.split()) for text in examples[text_column_name]]
        return tokenizer(
            examples["text"],
            padding=padding,
            truncation=True,
            max_length=data_args.max_seq_length,
            return_special_tokens_mask=True,
        )

    tokenized_datasets = filtered_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=not data_args.overwrite_cache,
        remove_columns=column_names
    )

    # Data collator
    # This one will take care of randomly masking the tokens.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, emo_mlm_probability=data_args.emo_mlm_probability, mlm_probability=data_args.mlm_probability, emo_lexicon=emo_lexicon)

     # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"] if training_args.do_train else None,
        eval_dataset=tokenized_datasets["validation"] if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

     # Training
    if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if (model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path))
            else None
        )
        train_result = trainer.train(model_path=model_path)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(train_result.metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))


if __name__ == "__main__":
    main()