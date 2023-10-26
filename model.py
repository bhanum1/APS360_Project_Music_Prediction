import torch #pytorch
import torch.nn as nn
from torch.autograd import Variable


PATH = '/Users/bhanumamillapalli/Documents/GitHub/APS360_Project_Music_Prediction'

# Load all the data and convert to Variable form
train_data = Variable(torch.load(PATH + '/train_data.pt'))
train_labels = Variable(torch.load(PATH + '/val_data.pt'))
val_data = Variable(torch.load(PATH + '/val_data.pt'))
val_labels = Variable(torch.load(PATH + '/val_labels.pt'))
test_data = Variable(torch.load(PATH + '/test_data.pt'))
test_labels = Variable(torch.load(PATH + '/test_labels.pt'))


#reshaping to rows, timestamps, features
X_train = torch.reshape(train_data,   (train_data.shape[0], 1, train_data.shape[1], train_data.shape[2]))
X_val = torch.reshape(val_data,  (val_data.shape[0], 1, val_data.shape[1], val_data.shape[2]))
X_test = torch.reshape(test_data,  (test_data.shape[0], 1, test_data.shape[1], test_data.shape[2]))

class MusicNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers):
        super(MusicNet, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        
    def forward(self, x, hidden):
        batch_size = x.size(0)
        x = x.long()
        lstm_out, hidden = self.lstm(x, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        
        out = lstm_out.view(batch_size, -1)
        out = out[:,-1]
        return out, hidden
    
    def init_hidden(self, batch_size):
        hidden_state = torch.randn(self.n_layers, batch_size, self.hidden_dim)
        cell_state = torch.randn(self.n_layers, batch_size, self.hidden_dim)
        hidden = (hidden_state, cell_state)
        return hidden
    




model = MusicNet(3, 3, 1)

batch_size = 10
epochs = 2
counter = 0
print_every = 1000
clip = 5

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

model.train()
for i in range(epochs):
    h = model.init_hidden(batch_size)
    
    for index in range(len(train_labels)):
        h = tuple([e.data for e in h])
        model.zero_grad()

        inputs = X_train[batch_size * (index-1):batch_size*index,:,:,:]
        labels = train_labels[batch_size * (index-1):batch_size*index]
        
        output, h = model(inputs, h)
        loss = criterion(output.squeeze(), labels.float())
        print("Training Loss:", loss)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
 