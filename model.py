import torch #pytorch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data
import torch.nn.functional as F

PATH = '/Users/bhanumamillapalli/Documents/GitHub/APS360_Project_Music_Prediction'

# Load all the data and convert to Variable form
train_data = torch.load(PATH + '/train_data.pt')
train_labels = torch.load(PATH + '/train_labels.pt')
val_data = torch.load(PATH + '/val_data.pt')
val_labels = torch.load(PATH + '/val_labels.pt')
test_data = torch.load(PATH + '/test_data.pt')
test_labels = torch.load(PATH + '/test_labels.pt')


#reshaping to rows, timestamps, features
#X_train = torch.reshape(train_data,   (train_data.shape[0], 1, train_data.shape[1], train_data.shape[2]))
#X_val = torch.reshape(val_data,  (val_data.shape[0], 1, val_data.shape[1], val_data.shape[2]))
#X_test = torch.reshape(test_data,  (test_data.shape[0], 1, test_data.shape[1], test_data.shape[2]))

class MusicNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers):
        super(MusicNet, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.linear = nn.Linear(3,3)
        
    def forward(self, x):
        lstm_out, hidden = self.lstm(x)
        output = F.relu(self.linear(hidden[0]))
        return output


input_dim = 3
hidden_dim = 3
n_layers = 1

model = MusicNet(input_dim, hidden_dim, n_layers)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
epochs = 20

X_train = train_data[:1000]
train_labels = train_labels[:1000]

train_loader = data.DataLoader(data.TensorDataset(X_train, train_labels), shuffle=True, batch_size=20)

for i in range(epochs):
    model.train()
    for X_batch,y_batch in train_loader:
        output = model(X_batch)
        loss = criterion(output, y_batch)
        print('Testing Loss: ', loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
