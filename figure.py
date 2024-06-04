import torch
import matplotlib.pyplot as plt
import numpy as np
PATH = '/Users/bhanumamillapalli/Documents/GitHub/APS360_Project_Music_Prediction/data/'




data= torch.load(PATH + '/train_data.pt')

print(data.shape)



'''
pitches = data[:,:,0]
starts = data[:,:,1]
durations = data[:,:,2]


pitches = pitches.reshape(-1)
plt.hist(pitches, color = 'xkcd:off blue')
plt.title("Distribution of Pitches")
plt.xlabel("Pitch Value")
plt.ylabel("Count")
plt.show()

starts = starts.reshape(-1)
plt.hist(starts, color = 'xkcd:off blue')
plt.title("Distribution of Start Times")
plt.xlabel("Start Time")
plt.ylabel("Count")
plt.show()
'''

durations = durations.reshape(-1)
plt.hist(durations, color = 'xkcd:off blue')
plt.title("Normalized Distribution of Durations")
plt.xlabel("Duration")
plt.ylabel("Count")
plt.show()
