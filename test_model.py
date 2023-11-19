# Import data
import torch #pytorch
import numpy as np
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import matplotlib.pyplot as plt
PATH = '/Users/bhanumamillapalli/Documents/GitHub/APS360_Project_Music_Prediction'
torch.manual_seed(100)

# Load all the data
train_data = torch.load(PATH + '/train_data.pt')
train_labels = torch.load(PATH + '/train_labels.pt')
val_data = torch.load(PATH + '/val_data.pt')
val_labels = torch.load(PATH + '/val_labels.pt')


# Batch the data
batch_size = 264
#batch_size = 4
pitch_train_loader = data.DataLoader(data.TensorDataset(train_data, train_labels[:,0]), shuffle=True, batch_size = batch_size)
pitch_val_loader = data.DataLoader(data.TensorDataset(val_data, val_labels[:,0]), shuffle=True, batch_size = batch_size)


# Define the model
class RNN(nn.Module):
  def __init__(self, vocab_size, hidden_size, n_layers=1):
    super(RNN, self).__init__()

    # one-hot
    self.ident = torch.eye(vocab_size)

    self.rnn = nn.LSTM(vocab_size + 2, hidden_size, n_layers, batch_first=True)

    # classifier
    self.fc = nn.Linear(hidden_size, 88)

  def forward(self, x):
    m = nn.Dropout(p=0.2)
    one_hot = []
    for seq in x:
      seq_pitch = seq[:,0]
      seq_pitch = seq_pitch.type(torch.int64)
      item = self.ident[seq_pitch]
      item = torch.cat((item, seq[:,1:]), dim=1)
      one_hot.append(item)

    input = torch.stack(one_hot)
    print(input[0,0,:])
    out, _ = self.rnn(m(input))
    out = self.fc(out[:,-1,:])
    
    return out

test = RNN(88,11)

from matplotlib import pyplot as plt

optimizer = torch.optim.Adam(test.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

train_loss, val_loss = [], []

for epoch in range(20):
    total_train_loss = 0
    for X_batch, y_batch in pitch_train_loader:
        optimizer.zero_grad()
        pred = test(X_batch)
        y_batch = y_batch.long()
        loss = criterion(pred, y_batch)
        total_train_loss += loss.detach().numpy()
        loss.backward()
        optimizer.step()

    train_loss.append(total_train_loss)

    total_val_loss = 0
    for X_batch, y_batch in pitch_val_loader:
        pred = test(X_batch)
        y_batch = y_batch.long()
        loss = criterion(pred, y_batch)
        total_val_loss += loss.detach().numpy()

    val_loss.append(total_val_loss)
    print("Epoch %d; Train Loss %f; Val Loss %f" % (
            epoch+1, train_loss[-1], val_loss[-1]))


#print(torch.argmax(F.softmax(test(train_data[0:batch_size]), dim = 1), dim=1))
#print(train_labels[:,0][0:batch_size])

# Validation accuracy
pred = torch.argmax(F.softmax(test(val_data), dim = 1), dim=1)
correct = 0
correct_preds = []
for index in range(len(pred)):
    if val_labels[:,0][index] == pred[index]:
        correct+=1
        correct_preds.append(int(pred[index]))


print(correct/len(val_labels))
#print(correct_preds)