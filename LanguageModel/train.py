import torch
import torch.nn as nn
import torch.nn.functional as F

from model import WordLSTM
from data import load_data, get_batches


x_int, y_int, vocab_size = load_data()

net = WordLSTM(vocab_size=vocab_size)

def train(net, epochs=10, batch_size=32, lr=3e-3, clip=1, print_every=256):
    
    # Optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    
    # Loss
    criterion = nn.CrossEntropyLoss()
    
    counter = 0
    net.train()
    for e in range(epochs):
        # Init hidden state
        h = net.init_hidden(batch_size)
        
        for x, y in get_batches(x_int, y_int, batch_size):
            counter += 1
            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)
            h = tuple([each.data for each in h])
            
            net.zero_grad()
            output, h = net(inputs, h)
            
            loss = criterion(output, targets.view(-1))
            
            loss.backward()
            
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            
            optimizer.step()
            
            if counter % print_every == 0:
                print(f'Epoch: {e+1}/{epochs}',
                      f'Step: {counter}...')

if __name__ == "__main__":
    train(net)
    torch.save(net, '.data/model.pth')
