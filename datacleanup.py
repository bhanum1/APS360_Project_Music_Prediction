import numpy as np
import torch
import csv
import pretty_midi
import random

from sklearn.preprocessing import MinMaxScaler

# Get data into list
PATH = '/Users/bhanumamillapalli/Documents/Python/APS360_Project'
data = []
with open(PATH + '/maestro-v3.0.0/maestro-v3.0.0.csv') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        data.append(row[4])


data_points = []
labels = []
# Convert sample file into pm object
sample_size = 100

#generate empty lists
pitches = []
starts = []
durations = []


test = []
i = 0
for file in data[1:]:
    i+=1
    if i%1000 == 0:
        print(i)
    file = PATH + '/maestro-v3.0.0/'+ file
    pm = pretty_midi.PrettyMIDI(file)

    instrument = pm.instruments[0]

    #generate set of indices of random notes
    indices = random.sample(range(0,len(instrument.notes)-(sample_size+ 1)),10)

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



scaler = MinMaxScaler(feature_range=(-1, 1), copy=True)
normalized_starts = scaler.fit_transform(np.array(starts).reshape(-1,1))
normalized_durations = scaler.fit_transform(np.array(durations).reshape(-1,1))

notes = []
for i in range(len(pitches)):
    if (i+1) % (sample_size+1) == 0:
        data_points.append(notes)
        notes = []
        labels.append([pitches[i], normalized_starts[i][0], normalized_durations[i][0]])
    else:
        notes.append([pitches[i], normalized_starts[i][0], normalized_durations[i][0]])
    
dataset = torch.FloatTensor(data_points)
labels = torch.FloatTensor(labels)



torch.save(dataset, PATH + '/dataset.pt')
torch.save(labels, PATH + '/labels.pt')

with open(PATH + '/dataset.csv', 'w') as f:
     
    # using csv.writer method from CSV package
    write = csv.writer(f)
    write.writerows(data_points)

with open(PATH +'/labels.csv', 'w') as f:
     
    # using csv.writer method from CSV package
    write = csv.writer(f)
    write.writerows(labels)
