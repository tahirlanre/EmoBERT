# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
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
""" Finetuning the library models for sequence classification on GLUE."""
# Adapted from transformers/examples/text-classification/run_glue.py

import logging
import os
import sys
import glob

import numpy as np
from datasets import load_dataset

import transformers
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    Trainer
)

import torch 
import torch.nn.functional as F

logger = logging.getLogger(__name__)

def model_fn(model_dir):
    """
    Loads model and its weights,and return the loaded model
    for inference
    Args:
        model_dir: the path to the S3 bucket containing the model file
    Returns:
        a loaded model transferred to the appropriate device
    """
    logger.debug("in model_fn()")

    #load BERT initial model
    model = BertForSequenceClassification.from_pretrained(model_dir)
    return model

def predict_fn(dataset, model, tokenizer):
    model.eval()
    predictions = []
    
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer
    )

    result = trainer.predict(test_dataset=dataset)
    predictions = result.predictions            
    return predictions

def main():
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    datasets = load_dataset('csv', data_files={'test':['data/pre_covid.csv']})
    datasets = datasets['test']

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples['processed_text'],)
        )
        result = tokenizer(*args, 
                        padding='max_length',
                        max_length=128,
                        truncation=True)
        return result

    tokenized_datasets = datasets.map(preprocess_function, batched=True, remove_columns=['id', 'processed_text'])

    model_dirs = glob.glob('saved_output/cls_loss/*/*_emo_wp_mlm_april')
    models = {}
    logging.info(f'No of models found {len(model_dirs)}')
    if len(model_dirs) > 1:
        logging.info(f'**** Initiating model classes ****')
        for model_dir in model_dirs:
            cat = model_dir.split('/')[-1].split('_')[0]
            if cat not in models.keys():
                models[cat] = [model_fn(model_dir)]
            else:
                models[cat].append(model_fn(model_dir))
    else:
        raise ValueError("Please provide location of saved models")

    for cat, model in models.items():
        predictions = []
        for i, m in enumerate(model):
            logging.info(f'**** Predicting  model no: {i} for {cat} class ****')
            output = predict_fn(tokenized_datasets, m, tokenizer)
            output = F.softmax(torch.from_numpy(output))
            predictions.append(output)
            torch.cuda.empty_cache()

        predictions = sum(predictions)/len(predictions)
        predictions = np.argmax(predictions, axis=1)

        output_pred_file = f'pre_covid_{cat}.txt'

        with open(output_pred_file, "w") as writer:
            logger.info(f"***** Test results *****")
            writer.write("index\tprediction\n")
            for index, item in enumerate(predictions):
        #         item = label_list[item]
                writer.write(f"{index}\t{item}\n")

if __name__ == "__main__":
    main() 