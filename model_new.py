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



# Normalize training & validation data
X_train = torch.nn.functional.normalize(train_data, p=2.0, dim=(1,2), eps=1e-12, out=None)
X_val = torch.nn.functional.normalize(val_data, p=2.0, dim=(1,2), eps=1e-12, out=None)
y_train = torch.nn.functional.normalize(train_labels, p=2.0, dim=1, eps=1e-12, out=None)
y_val = torch.nn.functional.normalize(val_labels, p=2.0, dim=1, eps=1e-12, out=None)

#
# Ground truth processing
#

# One-hot encoding for pitch labels
# 88 notes on piano, subtract 21 from all labels to get from range of 0-87
y_train_pitch  = train_labels[:,0]
for data_point in y_train_pitch:
    data_point -= 21

y_val_pitch  = val_labels[:,0]
for data_point in y_val_pitch:
    data_point -= 21


# Batch the data
batch_size = 264
pitch_train_loader = data.DataLoader(data.TensorDataset(X_train, y_train_pitch), shuffle=True, batch_size = batch_size)
note_start_train_loader = data.DataLoader(data.TensorDataset(X_train, y_train[:,1]), shuffle=True, batch_size = batch_size)
duration_train_loader = data.DataLoader(data.TensorDataset(X_train, y_train[:,2]), shuffle=True, batch_size = batch_size)
pitch_val_loader = data.DataLoader(data.TensorDataset(X_val, y_val_pitch), shuffle=True, batch_size = 2)
note_start_val_loader = data.DataLoader(data.TensorDataset(X_val, y_val[:,1]), shuffle=True, batch_size = 2)
duration_val_loader = data.DataLoader(data.TensorDataset(X_val, y_val[:,2]), shuffle=True, batch_size = 2)


# Define pitch network
class PitchNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers = 1):
        super(PitchNet, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
    
    def forward(self, x, h):
        lstm_out, hidden = self.lstm(x,h)
        function = nn.LogSoftmax(dim=2)
        output = function(hidden[0])
        pred = torch.argmax(F.softmax(hidden[0], dim=2),dim=2)
        return output, pred, lstm_out
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())

        return hidden

# Define NoteStart Network
class NoteStartNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers = 1):
        super(NoteStartNet, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.fc1 = nn.Linear(self.hidden_dim, 24)
        self.fc2 = nn.Linear(24, 1)
    
    def forward(self, x, h):
        lstm_out, hidden = self.lstm(x,h)
        output = F.relu(self.fc1(hidden[0]))
        output = F.relu(self.fc2(output))

        return output
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())

        return hidden

# Define Duration Network
class DurationNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers = 1):
        super(DurationNet, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.fc1 = nn.Linear(self.hidden_dim, 24)
        self.fc2 = nn.Linear(24, 1)
    
    def forward(self, x, h):
        lstm_out, hidden = self.lstm(x,h)
        output = F.relu(self.fc1(hidden[0]))
        output = F.relu(self.fc2(output))

        return output
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())

        return hidden




'''
#Create Pitch Network
input_dim = 3
hidden_dim = 88

model = PitchNet(input_dim, hidden_dim)

criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
epochs = 500

#Train Pitch Network
loss_values = []
for i in range(epochs):
    #print(i)
    model.train()
    h = model.init_hidden(batch_size)

    # Training
    for X_batch,y_batch in pitch_train_loader:
        h = tuple([e.data for e in h])
        output,pred,lstm_out = model.forward(X_batch,h)

        output = torch.squeeze(output)
        y_batch = y_batch.long()


        loss = criterion(output, y_batch)
        loss_values.append(loss.detach().numpy())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

plt.plot(loss_values)
#plt.show()


# Validation

    model.eval()
    h = model.init_hidden(2)
    val_loss = 0
    for X_point,y_point in pitch_val_loader:
        h = tuple([e.data for e in h])
        output, pred = model.forward(X_point,h)

        output = torch.squeeze(output)
        y_point = y_point.long()


        loss = criterion(output, y_point)
        val_loss += loss.detach().numpy()
    loss_values.append(val_loss)

plt.plot(loss_values)
plt.show()


model.eval()
h = model.init_hidden(2)
h = tuple([e.data for e in h])
output, pred,lstm_out = model.forward(X_train[0:2],h)
print(lstm_out)
print(pred, y_train_pitch[0:2])


# Create Note Start Network
input_dim = 3
hidden_dim = 12

model = NoteStartNet(input_dim, hidden_dim)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
epochs = 20

loss_values = []
val_loss_values = []
for i in range(epochs):
    print(i)
    model.train()
    h = model.init_hidden(batch_size)

    # Training
    for X_batch,y_batch in note_start_train_loader:
        h = tuple([e.data for e in h])
        output = model.forward(X_batch,h)

        output = torch.squeeze(output)
        loss = criterion(output, y_batch)
        loss_values.append(loss.detach().numpy())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    model.eval()
    h = model.init_hidden(2)
    val_loss = 0
    for X_point,y_point in note_start_val_loader:
        h = tuple([e.data for e in h])
        output= model.forward(X_point,h)

        output = torch.squeeze(output)
        loss = criterion(output, y_point)
        val_loss += loss.detach().numpy()
    val_loss_values.append(val_loss)

plt.plot(val_loss_values)
plt.show()
'''

# Create Duration Network
input_dim = 3
hidden_dim = 12

model = DurationNet(input_dim, hidden_dim)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
epochs = 5

loss_values = []
val_loss_values = []
for i in range(epochs):
    print(i)
    model.train()
    h = model.init_hidden(batch_size)

    # Training
    for X_batch,y_batch in duration_train_loader:
        h = tuple([e.data for e in h])
        output = model.forward(X_batch,h)

        output = torch.squeeze(output)
        loss = criterion(output, y_batch)
        loss_values.append(loss.detach().numpy())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    model.eval()
    h = model.init_hidden(2)
    val_loss = 0
    for X_point,y_point in duration_val_loader:
        print(X_point, y_point)
        h = tuple([e.data for e in h])
        output= model.forward(X_point,h)

        output = torch.squeeze(output)
        loss = criterion(output, y_point)
        val_loss += loss.detach().numpy()
    val_loss_values.append(val_loss)

plt.plot(val_loss_values)
plt.show()