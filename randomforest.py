import pandas as pd
import numpy
import torch
import csv
import random
from sklearn.model_selection import train_test_split

PATH = '/Users/bhanumamillapalli/Documents/Python/APS360_Project'

data = torch.load(PATH + '/dataset_final.pt')
pitch_data = data[:,:,0]
labels = torch.load(PATH + '/labels_final.pt')
pitch_labels = labels[:,0]

X = pd.DataFrame(pitch_data.numpy())
y = pd.DataFrame(pitch_labels.numpy())


#X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=101)
correct = 0

for i in range(len(X)):
    guess = max(set(X.loc[i].values[-30:]), key=list(X.loc[i][-30:]).count)
    if guess == y.values.ravel()[i]:
        correct += 1

print(correct/len(y))