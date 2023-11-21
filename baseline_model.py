import pandas as pd
import numpy
import torch
import csv
import random
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
import torch.utils.data as data
import torch.nn.functional as F
import matplotlib.pyplot as plt

PATH = '/Users/bhanumamillapalli/Documents/GitHub/APS360_Project_Music_Prediction'
torch.manual_seed(100)

# Load all the data
train_data = torch.load(PATH + '/data/train_data.pt')
train_labels = torch.load(PATH + '/data/train_labels.pt')
val_data = torch.load(PATH + '/data/val_data.pt')
val_labels = torch.load(PATH + '/data/val_labels.pt')

train_pitches = train_data[:,:,0]
train_starts = train_data[:,:,1]
train_durations = train_data[:100,:,2]
val_pitches = val_data[:,:,0]
val_starts = val_data[:,:,1]
val_durations = val_data[:,:,2]

'''
#Baseline Random Forest for pitch
clf = RandomForestClassifier()
clf.fit(train_pitches, train_labels[:,0])
print("Forest Classifier Pitch Accuracy: ", clf.score(val_pitches, val_labels[:,0]))

#Baseline SVM for pitch
clf = svm.SVC()
clf.fit(train_pitches, train_labels[:,0])
print("SVM Classifier Pitch Accuracy: ", clf.score(val_pitches, val_labels[:,0]))





# Baseline Random Forest for start
reg = RandomForestRegressor()
reg.fit(train_starts, train_labels[:,1])
pred = reg.predict(val_starts)

loss = 0
for i in range(len(pred)):
    loss += (val_labels[:,1][i] - pred[i])**2 #MSE
print("Forest Regressor Note Start Loss: ", loss)

# Baseline Random Forest for duration
reg = RandomForestRegressor()
reg.fit(train_durations, train_labels[:,2])
pred = reg.predict(val_durations)

loss = 0
for i in range(len(pred)):
    loss += (val_labels[:,2][i] - pred[i])**2 #MSE

print("Forest Regressor Note Duration Loss: ", loss)

# Baseline SVM for start
reg = svm.SVR()
reg.fit(train_starts, train_labels[:,1])
pred = reg.predict(val_starts)

loss = 0
for i in range(len(pred)):
    loss += (val_labels[:,1][i] - pred[i])**2 #MSE

print("SVM Regressor Note Start Loss: ", loss)

'''

# Baseline SVM for duration
reg = RandomForestRegressor()
print('fitting...')
reg.fit(train_durations, train_labels[:100,2])
print('fit done')
pred = reg.predict(val_durations)

loss = 0
print(len(pred))
for i in range(len(pred)):
    loss += (val_labels[:,2][i] - pred[i])**2 #MSE
print("Forest Regressor Note Duration Loss: ", loss)
