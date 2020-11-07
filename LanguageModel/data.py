import re
import pickle
import numpy as np

# Load data
datapath = './data/plots_text.pickle'
pickle_in = open(datapath, 'rb')
movie_plots = pickle.load(pickle_in)

# Clean data
movie_plots = [re.sub("[^a-z' ]", "", i) for i in movie_plots]

# Create sequences of length 5
def create_seq(text, seq_len=5):
    sequences = []
    if len(text.split()) > seq_len:
        for i in range(seq_len, len(text.split())):
            seq = text.split()[i-seq_len:i+1]
            sequences.append(' '.join(seq))
        return sequences
    else:
        return [text]
    
seqs = [create_seq(i) for i in movie_plots]
seqs = sum(seqs, [])

# Inputs and targets
x = []
y = []
for s in seqs:
    x.append(' '.join(s.split()[:-1]))
    y.append(' '.join(s.split()[1:]))

# Word to integer values
int2token = {}
cnt = 0
for w in set(' '.join(movie_plots).split()):
    int2token[cnt] = w
    cnt += 1

token2int = {t: i for i, t in int2token.items()}

vocab_size = len(int2token)

# Getting integer sequences
def get_integer_seq(seq):
    return [token2int[w] for w in seq.split()]

def load_data():
    x_int = np.array([get_integer_seq(i) for i in x])
    y_int = np.array([get_integer_seq(i) for i in y])
    return x_int, y_int, vocab_size

# Form batches
def get_batches(x_int, y_int, batch_size):
    prv = 0
    for n in range(batch_size, x_int.shape[0], batch_size):
        x = x_int[prv:n, :]
        y = y_int[prv:n, :]
        prv = n
        yield x, y
