#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

from data import LabeledDataset
from model import LSTMTagger


def parse_args():
    p = ArgumentParser()
    p.add_argument('--train-file', type=str, required=True)
    p.add_argument('--batch-size', type=int, default=64)
    return p.parse_args()


def train(model, data_loader, epochs):
    opt = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    if use_cuda:
        criterion = criterion.cuda()
    for i in range(epochs):
        epoch_loss = 0
        print("Epoch {} / {}".format(i+1, epochs))
        for bi, (x, y) in enumerate(data_loader):
            x = Variable(x)
            y = Variable(y)
            if use_cuda:
                x = x.cuda()
                y = y.cuda()
            out, _ = model(x)
            opt.zero_grad()
            loss = criterion(out.view(-1, out.size(2)), y.view(-1))
            epoch_loss += loss.data[0]
            loss.backward()
            opt.step()
        print(epoch_loss / bi)

def main():
    args = parse_args()
    data = LabeledDataset(args.train_file)
    loader = DataLoader(data, batch_size=args.batch_size)
    model = LSTMTagger(len(data.x_vocab), len(data.y_vocab), 20, 64, 1)
    if use_cuda:
        model = model.cuda()
    train(model, loader, epochs=2)

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    use_cuda = False
    main()
