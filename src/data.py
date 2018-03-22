#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

from sys import stdin
import gzip
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader


class LabeledDataset(Dataset):
    def __init__(self, stream_or_file=None):
        super().__init__()
        self.x_vocab = {'PAD': 0}
        self.y_vocab = {'PAD': 0}
        self._load_stream_or_file(stream_or_file)

    def _load_stream_or_file(self, stream_or_file):
        if stream_or_file is not None:
            if isinstance(stream_or_file, str):
                if stream_or_file.endswith('.gz'):
                    with gzip.open(stream_or_file, 'rt') as stream:
                        self.load_data(stream)
                else:
                    with open(stream_or_file) as stream:
                        self.load_data(stream)
            else:
                self.load_data(stream_or_file)

    def load_data(self, stream):
        samples = [line.strip().split('\t')[:2] for line in stream]
        self.samples = [s for s in samples if len(s) >= 2 and
                        len(s[0]) == len(s[1])]
        if len(self.samples) < len(samples):
            diff = len(samples) - len(self.samples)
            logging.warning("{} invalid samples were filtered".format(diff))
        self.maxlen = max(len(s[0]) for s in self.samples)
        self.create_matrices()

    def pad_sample(self, sample):
        return [0] * (self.maxlen - len(sample)) + sample

    def compute_label_no(self):
        """compute the number of labels.
        Padding is not included"""
        labels = set()
        for s in self.samples:
            labels |= set(s[1])
        # add one for padding
        self.n_labels = len(labels) + 1

    def create_matrices(self):
        self.compute_label_no()
        self.x = [
            self.pad_sample([self.x_vocab.setdefault(c, len(self.x_vocab))
                            for c in sample[0]]) for sample in self.samples
        ]
        y = []
        for sample in self.samples:
            sample = [self.y_vocab.setdefault(c, len(self.y_vocab))
                      for c in sample[1]]
            padded = self.pad_sample(sample)
            y.append(padded)
        self.x = np.array(self.x)
        self.y = np.array(y)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def main():
    data = LabeledDataset(stdin)
    loader = DataLoader(data, batch_size=4)
    for i, batch in enumerate(loader):
        print(i, batch)

if __name__ == '__main__':
    main()
