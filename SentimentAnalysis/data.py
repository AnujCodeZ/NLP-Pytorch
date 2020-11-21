import os
import string

import torch
from torchtext import data


def load_file(filepath, device, MAX_VOCAB_SIZE=25000):
    tokenizer = lambda x: str(x).translate(str.maketrans('', '', string.punctuation)).strip().split()
    
    TEXT = data.Field(sequential=True, lower=True, tokenize=tokenizer, fix_length=100)
    LABEL = data.Field(sequential=False, use_vocab=False)
    
    print('Loading from csv...')
    tv_datafields = [('text', TEXT), ('label', LABEL)]
    
    train, valid, test = data.TabularDataset.splits(path=filepath,
                                                    train='Train.csv', validation='Valid.csv',
                                                    test='Test.csv', format='csv',
                                                    skip_header=True, fields=tv_datafields)
    
    print('Building vocab...')
    TEXT.build_vocab(train, max_size=MAX_VOCAB_SIZE)
    
    print('Constructing iterators...')
    train_iter = data.BucketIterator(train, batch_size=32, sort_key=lambda x: len(x.text), 
                                     sort_within_batch=False, repeat=False, device=device)
    valid_iter = data.BucketIterator(valid, batch_size=32, sort_key=lambda x: len(x.text), 
                                     sort_within_batch=False, repeat=False, device=device)
    test_iter = data.BucketIterator(test, batch_size=32, sort_key=lambda x: len(x.text),
                                     sort_within_batch=False, repeat=False, device=device)
    
    return TEXT, LABEL, train, valid, test, train_iter, valid_iter, test_iter
