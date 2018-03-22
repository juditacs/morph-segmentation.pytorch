#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMTagger(nn.Module):
    def __init__(self, input_size, output_size, embedding_size, hidden_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, embedding_size)
        nn.init.xavier_uniform(self.embedding.weight)
        self.cell = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, bidirectional=True)
        self.out_proj = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        embedded = self.embedding(input)
        output, hidden = self.cell(embedded)
        output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]
        output = self.out_proj(output)
        # output = F.softmax(output)
        return output, hidden
