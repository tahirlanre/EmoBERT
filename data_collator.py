# Copyright 2020 The HuggingFace Team. All rights reserved.
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

# Adapted from transformers/data/data_collator.py

import random
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

import torch

from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase

@dataclass
class DataCollatorForLanguageModeling:
    """
    Data collator used for language modeling. Inputs are dynamically padded to the maximum length of a batch if they
    are not all of the same length.
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        mlm (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to use masked language modeling. If set to :obj:`False`, the labels are the same as the
            inputs with the padding tokens ignored (by setting them to -100). Otherwise, the labels are -100 for
            non-masked tokens and the value to predict for the masked token.
        emo_mlm_probability (:obj:`float`, `optional`, defaults to 0.3):
            The probability with which to (randomly) mask emolex tokens in the input, when :obj:`mlm` is set to :obj:`True`.
        mlm_probability (:obj:`float`, `optional`, defaults to 0.15):
            The probability with which to (randomly) mask tokens in the input, when :obj:`mlm` is set to :obj:`True`.
        emo_lexicon (:obj:`list`, `optional`):
            The list of words in emolex.
    .. note::
        For best performance, this data collator should be used with a dataset having items that are dictionaries or
        BatchEncoding, with the :obj:`"special_tokens_mask"` key, as returned by a
        :class:`~transformers.PreTrainedTokenizer` or a :class:`~transformers.PreTrainedTokenizerFast` with the
        argument :obj:`return_special_tokens_mask=True`.
    """

    tokenizer: PreTrainedTokenizerBase
    emo_lexicon: list 
    mlm: bool = True
    emo_mlm_probability: float = 0.3
    mlm_probability: float = 0.15
    
    def __post_init__(self):
        if self.mlm and self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )
    
    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # Handle dict or lists with proper padding and conversion to tensor.
        batch = self.tokenizer.pad(examples, return_tensors="pt")

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch

    def mask_tokens(
        self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        if self.emo_lexicon is None:
            raise ValueError(
                "Provide list of words from emo lexicon to mask"
            )
        emolex_indices = [list(map(lambda x: 1 if x in self.emo_lexicon else 0,self.tokenizer.convert_ids_to_tokens(val))) for val in inputs]
        emolex_indices = torch.tensor(emolex_indices, dtype=torch.bool)

        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        emo_probability_matrix = torch.full(labels.shape, 0.0)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()              

        emo_probability_matrix.masked_fill_(emolex_indices, value=self.emo_mlm_probability)
        emo_masked_indices = torch.bernoulli(emo_probability_matrix).bool()

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & emo_masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        probability_matrix.masked_fill_(emolex_indices, value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        masked_indices = emo_masked_indices + masked_indices
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

@dataclass
class EmolexDataCollatorForLanguageModeling:
    """
    Data collator used for language modeling. Inputs are dynamically padded to the maximum length of a batch if they
    are not all of the same length.
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        emo_lexicon (:obj:`list`, `optional`):
            The list of words in emolex.
        emo_lexicon (:obj:`list`, `optional`):
            The vocabulary of emo lexicon
        mlm (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to use masked language modeling. If set to :obj:`False`, the labels are the same as the
            inputs with the padding tokens ignored (by setting them to -100). Otherwise, the labels are -100 for
            non-masked tokens and the value to predict for the masked token.
        emo_mlm_probability (:obj:`float`, `optional`, defaults to 0.3):
            The probability with which to (randomly) mask emolex tokens in the input, when :obj:`mlm` is set to :obj:`True`.
        mlm_probability (:obj:`float`, `optional`, defaults to 0.15):
            The probability with which to (randomly) mask tokens in the input, when :obj:`mlm` is set to :obj:`True`.
    .. note::
        For best performance, this data collator should be used with a dataset having items that are dictionaries or
        BatchEncoding, with the :obj:`"special_tokens_mask"` key, as returned by a
        :class:`~transformers.PreTrainedTokenizer` or a :class:`~transformers.PreTrainedTokenizerFast` with the
        argument :obj:`return_special_tokens_mask=True`.
    """

    tokenizer: PreTrainedTokenizerBase
    emo_lexicon: list 
    emolex2id: dict
    mlm: bool = True
    emo_mlm_probability: float = 0.3
    mlm_probability: float = 0.15
    
    def __post_init__(self):
        if self.mlm and self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )
    
    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # Handle dict or lists with proper padding and conversion to tensor.
        batch = self.tokenizer.pad(examples, return_tensors="pt")

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch

    def mask_tokens(
        self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        # labels = inputs.clone()
        if self.emo_lexicon is None:
            raise ValueError(
                "Provide list of words from EmoLex to mask"
            )
        if self.emolex2id is None:
            raise ValueError(
                "Provide vocabulary for EmoLex"
            )
        labels = [list(map(lambda x: self.emolex2id[x] if x in self.emolex2id else 0, self.tokenizer.convert_ids_to_tokens(val)))for val in inputs]
        labels = torch.tensor(labels, dtype=torch.long)
        emolex_indices = [list(map(lambda x: 1 if x in self.emo_lexicon else 0,self.tokenizer.convert_ids_to_tokens(val))) for val in inputs]
        emolex_indices = torch.tensor(emolex_indices, dtype=torch.bool)

        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        emo_probability_matrix = torch.full(labels.shape, 0.0)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()              

        emo_probability_matrix.masked_fill_(emolex_indices, value=self.emo_mlm_probability)
        emo_masked_indices = torch.bernoulli(emo_probability_matrix).bool()

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & emo_masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        probability_matrix.masked_fill_(emolex_indices, value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        masked_indices = emo_masked_indices + masked_indices
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels