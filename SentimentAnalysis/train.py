import torch
import torch.nn as nn

from data import load_file
from model import SentimentModel


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TEXT, LABEL, train, valid, test, train_iter, valid_iter, test_iter = load_file(filepath='data/',
                                                                               device=device)

INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1

model = SentimentModel(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)

optimizer = torch.optim.SGD(model.parameters(), lr=3e-3)

criterion = nn.BCEWithLogitsLoss()

model = model.to(device)
criterion = criterion.to(device)

