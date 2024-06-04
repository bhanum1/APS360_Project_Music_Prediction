# Import data
import torch #pytorch
import numpy as np
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import matplotlib.pyplot as plt
PATH = '/Users/bhanumamillapalli/Documents/GitHub/APS360_Project_Music_Prediction'

#DO NOT CHANGE
torch.manual_seed(100)

# Load all the data
train_data = torch.load(PATH + '/data/train_data.pt')
train_labels = torch.load(PATH + '/data/train_labels.pt')
val_data = torch.load(PATH + '/data/val_data.pt')
val_labels = torch.load(PATH + '/data/val_labels.pt')

#################
#HYPERPARAMETERS#
#################

#pitch
pitch_network_type = 'GRU' #Either LSTM or GRU
pitch_hidden_size = 88
pitch_dropout_ratio = 0.2
pitch_learning_rate = 1E-4
pitch_epochs = 20
pitch_n_layers = 1
pitch_batch_size = 264

#start
start_dropout_ratio = 0.5
start_learning_rate = 1E-4
start_epochs = 20
start_batch_size = 264
start_layer1 = 20
start_layer2 = 10

#duration
duration_network_type = 'LSTM' #Either LSTM or GRU
duration_hidden_size = 88
duration_dropout_ratio = 0.2
duration_learning_rate = 1E-3
duration_epochs = 3
duration_n_layers = 1
duration_batch_size = 264

# Batch the data
pitch_train_loader = data.DataLoader(data.TensorDataset(train_data, train_labels[:,0]), shuffle=True, batch_size = pitch_batch_size)
pitch_val_loader = data.DataLoader(data.TensorDataset(val_data, val_labels[:,0]), shuffle=True, batch_size = pitch_batch_size)

start_train_loader = data.DataLoader(data.TensorDataset(train_data, train_labels[:,1]), shuffle=True, batch_size = start_batch_size)
start_val_loader = data.DataLoader(data.TensorDataset(val_data, val_labels[:,1]), shuffle=True, batch_size = start_batch_size)

duration_train_loader = data.DataLoader(data.TensorDataset(train_data, train_labels[:,2]), shuffle=True, batch_size = duration_batch_size)
duration_val_loader = data.DataLoader(data.TensorDataset(val_data, val_labels[:,2]), shuffle=True, batch_size = duration_batch_size)



# Define the pitch model
class Pitch_RNN(nn.Module):
  def __init__(self, hidden_size, vocab_size=88, n_layers=pitch_n_layers):
    super(Pitch_RNN, self).__init__()

    # one-hot
    self.ident = torch.eye(vocab_size)

    #rnn
    if pitch_network_type == 'LSTM':
        self.rnn = nn.LSTM(vocab_size + 2, hidden_size, n_layers, batch_first=True)
    elif pitch_network_type == 'GRU':
        self.rnn = nn.GRU(vocab_size + 2, hidden_size, n_layers, batch_first=True)
    else:
       print('Please choose a valid network type (GRU or LSTM)')

    # classifier
    self.fc = nn.Linear(hidden_size, 88)

  def forward(self, x):
    # define dropout
    m = nn.Dropout(p=pitch_dropout_ratio)

    # Convert data to proper input
    input = []
    for seq in x:
      seq_pitch = seq[:,0]
      seq_pitch = seq_pitch.type(torch.int64)
      item = self.ident[seq_pitch]

      # add the start time and duration to the one-hot matrix to form input
      item = torch.cat((item, seq[:,1:]), dim=1)
      input.append(item)
    
    input = torch.stack(input)
    #apply dropout
    out, _ = self.rnn(m(input))
    out = self.fc(out[:,-1,:])
    
    return out

# Define the start time model
class Start_ANN(nn.Module):
  def __init__(self, layer1 = start_layer1, layer2 = start_layer2):
    super(Start_ANN, self).__init__()

    # fully connected layers
    self.fc1 = nn.Linear(100, layer1)
    self.fc2 = nn.Linear(layer1, layer2)
    self.fc3 = nn.Linear(layer2, 1)

  def forward(self, x):
    # define dropout
    m = nn.Dropout(p=start_dropout_ratio)

    # take only the start and durations as input and flatten it
    input = x[:,:,1:].reshape(start_batch_size, 100)
    #apply dropout and all layers
    out = F.sigmoid(self.fc3(F.relu(self.fc2(F.relu(self.fc1(m(input)))))))
    
    return out

# Define the duration model
class Duration_ANN(nn.Module):
  def __init__(self, layer1 = start_layer1, layer2 = start_layer2):
    super(Duration_ANN, self).__init__()

    # fully connected layers
    self.fc1 = nn.Linear(100, layer1)
    self.fc2 = nn.Linear(layer1, layer2)
    self.fc3 = nn.Linear(layer2, 1)

  def forward(self, x):
    # define dropout
    m = nn.Dropout(p=duration_dropout_ratio)

    # take only the start and durations as input and flatten it
    input = x[:,:,1:].reshape(duration_batch_size, 100)
    #apply dropout and all layers
    out = F.sigmoid(self.fc3(F.relu(self.fc2(F.relu(self.fc1(m(input)))))))
    
    return out
#create networks
pitch_net = Pitch_RNN(pitch_hidden_size)
start_net = Start_ANN()
duration_net = Duration_ANN()


# Pitch training
optimizer = torch.optim.Adam(pitch_net.parameters(), lr=pitch_learning_rate)
criterion = nn.CrossEntropyLoss()

train_loss, val_loss, train_acc, val_acc = [], [], [], []

for epoch in range(pitch_epochs):

    #training data
    pitch_net.train()
    total_train_loss = 0
    num_examples = 0
    for X_batch, y_batch in pitch_train_loader:
        optimizer.zero_grad()
        pred = pitch_net(X_batch)
        y_batch = y_batch.long()
        loss = criterion(pred, y_batch)
        total_train_loss += loss.detach().numpy()
        num_examples += len(y_batch)
        loss.backward()
        optimizer.step()

    train_loss.append(total_train_loss/num_examples)
    

    #validation data
    pitch_net.eval()
    total_val_loss = 0
    num_examples = 0
    for X_batch, y_batch in pitch_val_loader:
        pred = pitch_net(X_batch)
        y_batch = y_batch.long()
        loss = criterion(pred, y_batch)
        total_val_loss += loss.detach().numpy()
        num_examples += len(y_batch)

    val_loss.append(total_val_loss/num_examples)
    
    '''
    # Training & validation accuracy
    pred = torch.argmax(F.softmax(pitch_net(train_data), dim = 1), dim=1)
    train_correct = 0
    for index in range(len(pred)):
        if train_labels[:,0][index] == pred[index]:
            train_correct+=1

    pred = torch.argmax(F.softmax(pitch_net(val_data), dim = 1), dim=1)
    val_correct = 0
    for index in range(len(pred)):
        if val_labels[:,0][index] == pred[index]:
            val_correct+=1

    train_acc.append(train_correct/len(train_data))
    val_acc.append(val_correct/len(val_data))
    '''
    print("Epoch %d; Train Loss %f; Val Loss %f; Train Acc %f; Val Acc %f" % (
            epoch+1, train_loss[-1], val_loss[-1], 0, 0))

'''
# start time training
optimizer = torch.optim.Adam(start_net.parameters(), lr=start_learning_rate)
criterion = nn.MSELoss()

train_loss, val_loss, = [], []

for epoch in range(start_epochs):
    #training data
    start_net.train()
    total_train_loss = 0
    num_examples = 0
    for X_batch, y_batch in start_train_loader:
        if X_batch.shape[0] == start_batch_size:
            optimizer.zero_grad()
            pred = start_net(X_batch).squeeze()
            loss = criterion(pred, y_batch) * 1E6
            total_train_loss += loss.detach().numpy()
            num_examples += len(y_batch)
            loss.backward()
            optimizer.step()

    train_loss.append(total_train_loss/num_examples)
    
    #validation data
    start_net.eval()
    total_val_loss = 0
    num_examples = 0
    for X_batch, y_batch in start_val_loader:
        if X_batch.shape[0] == start_batch_size:
            pred = start_net(X_batch).squeeze()
            loss = criterion(pred, y_batch) * 1E6
            total_val_loss += loss.detach().numpy()
            num_examples += len(y_batch)

    val_loss.append(total_val_loss/num_examples)

    print("Epoch %d; Train Loss %f; Val Loss %f" % (
            epoch+1, train_loss[-1], val_loss[-1]))

# duration training
optimizer = torch.optim.Adam(duration_net.parameters(), lr=duration_learning_rate)
criterion = nn.MSELoss()

train_loss, val_loss, = [], []

for epoch in range(duration_epochs):
    #training data
    duration_net.train()
    total_train_loss = 0
    num_examples = 0
    for X_batch, y_batch in duration_train_loader:
        if X_batch.shape[0] == start_batch_size:
            optimizer.zero_grad()
            pred = duration_net(X_batch).squeeze()
            loss = criterion(pred, y_batch) * 1E6
            total_train_loss += loss.detach().numpy()
            num_examples += len(y_batch)
            loss.backward()
            optimizer.step()

    train_loss.append(total_train_loss/num_examples)
    
    #validation data
    duration_net.eval()
    total_val_loss = 0
    num_examples = 0
    for X_batch, y_batch in duration_val_loader:
        if X_batch.shape[0] == start_batch_size:
            pred = start_net(X_batch).squeeze()
            loss = criterion(pred, y_batch) * 1E6
            total_val_loss += loss.detach().numpy()
            num_examples += len(y_batch)

    val_loss.append(total_val_loss/num_examples)

    print("Epoch %d; Train Loss %f; Val Loss %f" % (
            epoch+1, train_loss[-1], val_loss[-1]))


pred = []
total_loss = 0
for X_batch, y_batch in start_val_loader:
        if X_batch.shape[0] == start_batch_size:
            pred = start_net(X_batch).squeeze()
            loss = criterion(pred, y_batch)
            total_loss += loss.detach().numpy()

print("Validation Start Loss: ", total_loss)


from sklearn.ensemble import RandomForestRegressor

reg = RandomForestRegressor()
print('fitting...')
reg.fit(train_data[:100,1], train_labels[:100,1])
print('fit done')
pred = reg.predict(val_data[:,1])

loss = 0
for i in range(len(pred)):
    loss += (val_labels[:,1][i] - pred[i])**2 #MSE
print("Forest Regressor Note Duration Loss: ", loss)
'''