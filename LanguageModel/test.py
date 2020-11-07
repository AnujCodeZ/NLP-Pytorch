import random
import numpy as np
import torch
import torch.nn.functional as F

from data import token2int, int2token


net = torch.load('./data/model.pth', map_location='cpu')

def predict(net, tkn, h=None):
    
    x = np.array([[token2int[tkn]]])
    inputs = torch.from_numpy(x)
    h = tuple([each.data for each in h])
    out, h = net(inputs, h)
    
    p = F.softmax(out, dim=1).data
    p = p.numpy()
    p = p.reshape(p.shape[1], )
    
    top_n_idx = p.argsort()[-3:][::-1]
    sampled_token_index = top_n_idx[random.sample([0, 1, 2], 1)[0]]
    
    return int2token[sampled_token_index], h

def sample(net, size, prime='it is'):
    net.eval()
    h = net.init_hidden(1)
    toks = prime.split()
    
    for t in prime.split():
        token, h = predict(net, toks[-1], h)
        
    toks.append(token)
    
    for i in range(size-1):
        token, h = predict(net, toks[-1], h)
        toks.append(token)
    
    return ' '.join(toks)

print(sample(net, 5, 'how are'))