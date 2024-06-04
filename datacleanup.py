import numpy as np
import torch
import csv
import pretty_midi
import random
import pandas
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Get data into list
PATH = '/Users/bhanumamillapalli/Documents/GitHub/APS360_Project_Music_Prediction'
data = []
with open(PATH + '/maestro-v3.0.0/maestro-v3.0.0.csv') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        data.append(row[4])


data_points = []
unnorm_data_points = []
labels = []
unnorm_labels = []
# Convert sample file into pm object
sample_size = 50
num_samples = 50

#generate empty lists
pitches = []
starts = []
durations = []


test = []
j = 0
for file in data[1:]:
    j+=1
    if j%10 ==0:
        print(j)
    file = PATH + '/maestro-v3.0.0/'+ file
    pm = pretty_midi.PrettyMIDI(file)

    instrument = pm.instruments[0]

    #generate set of indices of random notes
    try:
        indices = random.sample(range(0,len(instrument.notes)-(sample_size+ 1)),num_samples)
    except:
        indices = [0] * num_samples
        print('Short', len(instrument.notes))

    for index in indices:
        sample_pitches = []
        sample_starts = []
        sample_durations = []
        for i, note in enumerate(instrument.notes[index:index+(sample_size+1)]):
            note_name = pretty_midi.note_number_to_name(note.pitch)
            duration = round(note.end - note.start,3)
            start = round(note.start,3)
            
            # scale pitch to 0-87
            sample_pitches.append(note.pitch - 21)

            sample_starts.append(start)
            sample_durations.append(duration)

            test.append([note.pitch, start, duration])

        # Organize by start time per sample
        sample_starts, sample_pitches, sample_durations = (list(t) for t in zip(*sorted(zip(sample_starts, sample_pitches, sample_durations))))
        pitches.extend(sample_pitches)
        starts.extend(sample_starts)
        durations.extend(sample_durations)

scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
normalized_starts = scaler.fit_transform(np.array(starts).reshape(-1,1))
normalized_durations = scaler.fit_transform(np.clip(np.array(durations).reshape(-1,1), 0,1))


notes = []
unnorm_notes = []
for i in range(len(pitches)):
    if (i+1) % (sample_size+1) == 0:
        data_points.append(notes)
        unnorm_data_points.append(unnorm_notes)
        notes = []
        unnorm_notes = []
        labels.append([pitches[i], normalized_starts[i], normalized_durations[i]])
        unnorm_labels.append([pitches[i], starts[i], durations[i]])
    else:
        notes.append([pitches[i], normalized_starts[i], normalized_durations[i]])
        unnorm_notes.append([pitches[i], starts[i], durations[i]])

dataset = torch.FloatTensor(data_points)
labels = torch.FloatTensor(labels)
unnorm_dataset = torch.FloatTensor(unnorm_data_points)
unnorm_labels = torch.FloatTensor(unnorm_labels)

torch.save(dataset, PATH + '/data/dataset.pt')
torch.save(labels, PATH + '/data/labels.pt')
torch.save(unnorm_dataset, PATH + '/data/unnorm_dataset.pt')
torch.save(unnorm_labels, PATH + '/data/unnorm_labels.pt')




#if you need to write as a .csv
'''
with open(PATH + '/data/dataset.csv', 'w') as f:
     
    # using csv.writer method from CSV package
    write = csv.writer(f)
    write.writerows(data_points)

with open(PATH +'/data/labels.csv', 'w') as f:
     
    # using csv.writer method from CSV package
    write = csv.writer(f)
    write.writerows(labels)
'''

# Split the data into train/validation/test with 60/20/20 splits
X_train, X_test, y_train, y_test = train_test_split(dataset,labels,test_size=0.2,random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,test_size=0.25,random_state=1)

# Check the sizes to make sure they're right
print("Training Data: ", X_train.shape)
print("Training Labels: ", y_train.shape)
print("Validation Data: ", X_val.shape)
print("Validation Labels: ", y_val.shape)
print("Testing Data: ", X_test.shape)
print("Testing Labels: ", y_test.shape)

# Save all the info
torch.save(X_train, PATH + '/data/train_data.pt')
torch.save(y_train, PATH + '/data/train_labels.pt')
torch.save(X_val, PATH + '/data/val_data.pt')
torch.save(y_val, PATH + '/data/val_labels.pt')
torch.save(X_test, PATH + '/data/test_data.pt')
torch.save(y_test, PATH + '/data/test_labels.pt')
