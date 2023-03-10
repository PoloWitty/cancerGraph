# coding=utf-8

import os
import pdb
import copy
import torch
import torch.nn.functional as F
from torch import nn
import ipdb

from torch.nn import CrossEntropyLoss
from transformers import (
        BertConfig,
        BertModel,
        BertForTokenClassification,
)

class BERTMultiNER2(BertForTokenClassification):
    def __init__(self, config, num_labels=3):
        super(BERTMultiNER2, self).__init__(config)
        self.num_labels = num_labels # B, I, O
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        
        self.dise_classifier = torch.nn.Linear(config.hidden_size, self.num_labels) # disease
        self.chem_classifier = torch.nn.Linear(config.hidden_size, self.num_labels) # chemical
        self.gene_classifier = torch.nn.Linear(config.hidden_size, self.num_labels) # gene/protein
        self.spec_classifier = torch.nn.Linear(config.hidden_size, self.num_labels) # species

        self.dise_classifier_2 = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.chem_classifier_2 = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.gene_classifier_2 = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.spec_classifier_2 = torch.nn.Linear(config.hidden_size, config.hidden_size)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, entity_type_ids=None, word_mask=None):
        sequence_output = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, head_mask=None)[0]
        batch_size,max_len,feat_dim = sequence_output.shape
        sequence_output = self.dropout(sequence_output)
        if entity_type_ids[0][0].item() == 0:
            '''
            Raw text data with trained parameters
            '''
            dise_sequence_output = F.relu(self.dise_classifier_2(sequence_output)) # disease logit value
            chem_sequence_output = F.relu(self.chem_classifier_2(sequence_output)) # chemical logit value
            gene_sequence_output = F.relu(self.gene_classifier_2(sequence_output)) # gene/protein logit value
            spec_sequence_output = F.relu(self.spec_classifier_2(sequence_output)) # species logit value

            dise_logits = self.dise_classifier(dise_sequence_output) # disease logit value
            chem_logits = self.chem_classifier(chem_sequence_output) # chemical logit value
            gene_logits = self.gene_classifier(gene_sequence_output) # gene/protein logit value
            spec_logits = self.spec_classifier(spec_sequence_output) # species logit value

            # update logit and sequence_output
            sequence_output = dise_sequence_output + chem_sequence_output + gene_sequence_output + spec_sequence_output
            logits = (torch.nn.functional.softmax(dise_logits,dim=-1), torch.nn.functional.softmax(chem_logits,dim=-1), torch.nn.functional.softmax(gene_logits,dim=-1), torch.nn.functional.softmax(spec_logits,dim=-1))
        else:
            ''' 
            Train, Eval, Test with pre-defined entity type tags
            '''
            # make 1*1 conv to adopt entity type
            dise_idx = copy.deepcopy(entity_type_ids)
            chem_idx = copy.deepcopy(entity_type_ids)
            gene_idx = copy.deepcopy(entity_type_ids)
            spec_idx = copy.deepcopy(entity_type_ids)

            dise_idx[dise_idx != 1] = 0
            chem_idx[chem_idx != 2] = 0
            gene_idx[gene_idx != 3] = 0
            spec_idx[spec_idx != 4] = 0 # kind of starge? This will x4 on species type output, so i change it as follows

            dise_idx = torch.where(entity_type_ids==1,1,0)
            chem_idx = torch.where(entity_type_ids==2,1,0)
            gene_idx = torch.where(entity_type_ids==3,1,0)
            spec_idx = torch.where(entity_type_ids==4,1,0)
            dise_sequence_output = dise_idx.unsqueeze(-1) * sequence_output        
            chem_sequence_output = chem_idx.unsqueeze(-1) * sequence_output
            gene_sequence_output = gene_idx.unsqueeze(-1) * sequence_output
            spec_sequence_output = spec_idx.unsqueeze(-1) * sequence_output

            # F.tanh or F.relu
            dise_sequence_output = F.relu(self.dise_classifier_2(dise_sequence_output)) # disease logit value
            chem_sequence_output = F.relu(self.chem_classifier_2(chem_sequence_output)) # chemical logit value
            gene_sequence_output = F.relu(self.gene_classifier_2(gene_sequence_output)) # gene/protein logit value
            spec_sequence_output = F.relu(self.spec_classifier_2(spec_sequence_output)) # species logit value

            dise_logits = self.dise_classifier(dise_sequence_output) # disease logit value
            chem_logits = self.chem_classifier(chem_sequence_output) # chemical logit value
            gene_logits = self.gene_classifier(gene_sequence_output) # gene/protein logit value
            spec_logits = self.spec_classifier(spec_sequence_output) # species logit value

            # update logit and sequence_output
            sequence_output = dise_sequence_output + chem_sequence_output + gene_sequence_output + spec_sequence_output
            logits = dise_logits + chem_logits + gene_logits + spec_logits

        outputs = (logits, )
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
                return ((loss,) + outputs)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                return (loss)
        else:
            return logits, word_mask