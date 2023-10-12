import pandas as pd
import numpy
import torch
import csv
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split

PATH = '/Users/bhanumamillapalli/Documents/Python/APS360_Project'

data = torch.load(PATH + '/dataset_final.pt')
pitch_data = data[:,:,0]
labels = torch.load(PATH + '/labels_final.pt')
pitch_labels = labels[:,0]

# Get just the pitch data
X = pd.DataFrame(pitch_data.numpy())
y = pd.DataFrame(pitch_labels.numpy())


# Create a train test split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=100)



# Baseline heuristic model
correct = 0

for i in range(len(X_test)):
    guess = max(set(X_test.values[i][-30:]), key=list(X_test.values[i][-30:]).count)
    if guess == y_test.values.ravel()[i]:
        correct += 1
print('Heuristic Accuracy:', correct/len(y_test))

# Baseline Random Forest
clf = RandomForestClassifier()
clf.fit(X_train, y_train.values.ravel())
print('Random Forest Accuracy:', clf.score(X_test,y_test.values.ravel()))

# Baseline SVM Model
clf = svm.SVC()
clf.fit(X_train, y_train.values.ravel())
print('SVM Accuracy:', clf.score(X_test,y_test.values.ravel()))
