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

def binary_accuracy(preds, y):
    
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    
    return acc

def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for i, batch in enumerate(iterator):
        optimizer.zero_grad()

        predictions = model(batch.text).squeeze(1)

        loss = criterion(predictions, batch.label.float())
        acc = binary_accuracy(predictions, batch.label)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        if i % 200 == 199:
            print(f"[{i}/{len(iterator)}] : epoch_acc: {epoch_acc / len(iterator):.2f}")
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            # prediction [batch_size]
            predictions = model(batch.text).squeeze(1)
            
            loss = criterion(predictions, batch.label.float())
            
            acc = binary_accuracy(predictions, batch.label)
        
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            
    return epoch_loss / len(iterator),  epoch_acc / len(iterator)

N_epoches = 5

best_valid_loss = float('inf')

for epoch in range(N_epoches):
        
    train_loss, train_acc = train(model, train_iter, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iter, criterion)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'Sentiment-model.pt')
        
    print(f'Epoch:  {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain  Loss: {train_loss: .3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\tValid  Loss: {valid_loss: .3f} | Valid Acc: {valid_acc*100:.2f}%')

model.load_state_dict(torch.load('Sentiment-model.pt'))

test_loss, test_acc = evaluate(model, test_iter, criterion)

print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')