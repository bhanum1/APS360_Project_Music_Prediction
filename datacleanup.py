import numpy as np
import torch
import csv
import pretty_midi
import random

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
for file in data[1:]:
    file = PATH + '/maestro-v3.0.0/'+ file
    pm = pretty_midi.PrettyMIDI(file)

    instrument = pm.instruments[0]

    #generate set of indices of random notes
    indices = random.sample(range(0,len(instrument.notes)-(sample_size+ 1)),10)

    for index in indices:
        notes = []
        for i, note in enumerate(instrument.notes[index:index+(sample_size+1)]):
            note_name = pretty_midi.note_number_to_name(note.pitch)
            duration = round(note.end - note.start,3)
            start = round(note.start,3)
            
            if len(notes) == sample_size:
                labels.append([note.pitch,start,duration])
            else:
                notes.append([note.pitch,start,duration])

        data_points.append(notes)


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