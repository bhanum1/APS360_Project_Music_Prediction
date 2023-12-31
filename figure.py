import torch
import matplotlib.pyplot as plt
PATH = '/Users/bhanumamillapalli/Documents/GitHub/APS360_Project_Music_Prediction'

train_data = torch.load(PATH + '/train_data.pt')
train_starts = train_data[:,:,1]
train_labels = torch.load(PATH + '/train_labels.pt')
start_labels = train_labels[:,1]


i = 15
print(start_labels[i])
data = list(train_starts[i])
data.append(start_labels[i])
plt.plot(data)
plt.title("Note Starts Across Sequence")
plt.xlabel("Index")
plt.ylabel("Normalized Note Start Time")
plt.show()
