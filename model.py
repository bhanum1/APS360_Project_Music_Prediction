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

# There are 88 possible notes on a piano, ranging from pitch value of 21 to 108

# reshaping to rows, timestamps, features
# train = torch.reshape(train_data,   (train_data.shape[0], 1, train_data.shape[1], train_data.shape[2]))
# X_val = torch.reshape(val_data,  (val_data.shape[0], 1, val_data.shape[1], val_data.shape[2]))
# X_test = torch.reshape(test_data,  (test_data.shape[0], 1, test_data.shape[1], test_data.shape[2]))

class MusicNet(nn.Module):
    def __init__(self, input_dim, hidden_dim_pitch, n_layers = 1):
        super(MusicNet, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim_pitch = hidden_dim_pitch
        self.input_dim = input_dim

        self.lstm_pitch = nn.LSTM(input_dim, hidden_dim_pitch, n_layers, batch_first=True)
        
    def forward(self, x, h):
        lstm_out, hidden = self.lstm_pitch(x,h)
        output = F.softmax(hidden[0], dim=2)
        output = torch.argmax(output, dim=2)
        return (output)
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim_pitch).zero_(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim_pitch).zero_())

        return hidden
    
input_dim = 3
hidden_dim_pitch = 88
batch_size = 200
model = MusicNet(input_dim, hidden_dim_pitch)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
epochs = 20

X_train = train_data[:1000]
train_labels = train_labels[:1000]

X_train = torch.nn.functional.normalize(X_train, p=2.0, dim=(1,2), eps=1e-12, out=None)
train_labels = train_labels[:,0]

train_loader = data.DataLoader(data.TensorDataset(X_train, train_labels), shuffle=True, batch_size = batch_size)

for i in range(epochs):
    model.train()
    h = model.init_hidden(batch_size)

    for X_batch,y_batch in train_loader:
        h = tuple([e.data for e in h])
        output = model.forward(X_batch,h)
        #y_batch = torch.reshape(y_batch,   (1, y_batch.shape[0]))
        output = torch.squeeze(output)
        
        loss = criterion(output, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
model.eval()
h = model.init_hidden(20)
h = tuple([e.data for e in h])
output = model(train_data[0:20],h)

