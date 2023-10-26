import torch
from sklearn.model_selection import train_test_split

PATH = '/Users/bhanumamillapalli/Documents/Python/APS360_Project'

data = torch.load(PATH + '/dataset_final.pt')
labels = torch.load(PATH + '/labels_final.pt')

# Split the data into train/validation/test with 60/20/20 splits
X_train, X_test, y_train, y_test = train_test_split(data,labels,test_size=0.2,random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,test_size=0.25,random_state=1)

# Check the sizes to make sure they're right
print("Training Data: ", X_train.shape)
print("Training Labels: ", y_train.shape)
print("Validation Data: ", X_val.shape)
print("Validation Labels: ", y_val.shape)
print("Testing Data: ", X_test.shape)
print("Testing Labels: ", y_test.shape)

# Save all the info
torch.save(X_train, PATH + '/train_data.pt')
torch.save(y_train, PATH + '/train_labels.pt')
torch.save(X_val, PATH + '/val_data.pt')
torch.save(y_val, PATH + '/val_labels.pt')
torch.save(X_test, PATH + '/test_data.pt')
torch.save(y_test, PATH + '/test_labels.pt')