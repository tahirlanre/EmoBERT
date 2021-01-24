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
# Adapted from transformers/src/transformers/models/bert/modeling_bert.py
import pickle

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers import (
    BertForMaskedLM, 
    BertTokenizer,
    BertModel
)
from transformers.modeling_outputs import MaskedLMOutput

emo_lexicon = pickle.load(open('data/emolex_emo_words.pkl', 'rb'))

def get_emb_weights(bert_model='bert-base-uncased'):
    model = BertModel.from_pretrained(bert_model)
    tokenizer = BertTokenizer.from_pretrained(bert_model)
    emb_list = []
    for word in emo_lexicon:
        emb_list.append(torch.mean(model.embeddings.word_embeddings.weight[
                                       tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))].detach(),
                                   axis=0).numpy().tolist())
    return torch.tensor(emb_list)

class EmoBertForMaskedLM(BertForMaskedLM):
    def __init__(self, config, output_dim):
        super().__init__(config)
        self.output_dim = output_dim

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.output_dim), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def resize_embedding_and_fc(self, new_num_tokens):
        # Change the FC 
        old_fc = self.cls.predictions.decoder
        self.cls.predictions.decoder = self._get_resized_fc(old_fc, new_num_tokens)
        
        # Change the bias
        old_bias = self.cls.predictions.bias
        self.cls.predictions.bias = self._get_resized_bias(old_bias, new_num_tokens)
        
        # Change the embedding
        # self.resize_token_embeddings(new_num_tokens)

    def _get_resized_bias(self, old_bias, new_num_tokens):
        old_num_tokens = old_bias.data.size()[0]
        if old_num_tokens == new_num_tokens:
            return old_bias

        new_bias = nn.Parameter(torch.zeros(new_num_tokens))
        new_bias.to(old_bias.device)

        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
        new_bias.data[:num_tokens_to_copy] = old_bias.data[:num_tokens_to_copy]
        return new_bias

    def _get_resized_fc(self, old_fc, new_num_tokens):
        old_num_tokens, old_embedding_dim = old_fc.weight.size()
        if old_num_tokens == new_num_tokens:
            return old_fc

        new_fc = nn.Linear(old_embedding_dim, new_num_tokens)
        new_fc.to(old_fc.weight.device)

        self._init_weights(new_fc)

        init_emb = get_emb_weights()

        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
        new_fc.weight.data[1:num_tokens_to_copy, :] = init_emb
        return new_fc