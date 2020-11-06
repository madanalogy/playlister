import torch
import random
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import os
import glob
import librosa
import librosa.display
import numpy as np
import time
# Load the other neccesary libraries

class CNN(torch.nn.Module):
    
    # Our batch shape for input x is (1, 20, 1200)
    
    def __init__(self):
        super(CNN, self).__init__()
        
        # Input channels = 1, output channels = 128
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # Becomes 10 600
        self.conv2 = torch.nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=1)
        # Becomes 5 300 (has 64 output channels becaue of previous)
        self.conv3 = torch.nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding =1)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = torch.nn.Linear(64 * 2 * 150, 256)
        
        self.fc2 = torch.nn.Linear(256, 128)

        self.fc3 = torch.nn.Linear(128, 64)

        # 64 input features, 10 output features for our 10 defined classes
        self.fc4 = torch.nn.Linear(64, 10)
        
    def forward(self, x):
        
        # Computes the activation of the first convolution
        # Size changes from (3, 32, 32) to (18, 32, 32)
        # print(x.shape)
        # print('first convolution')
        x = F.relu(self.conv1(x))
        # print(x.shape)
        # Size changes from (18, 32, 32) to (18, 16, 16)
        x = self.pool(x)
        # print(x.shape)
        
        # Reshape data to input to the input layer of the neural net
        # Size changes from (18, 16, 16) to (1, 4608)
        # Recall that the -1 infers this dimension from the other given dimension
        # print('second convolution')
        x = F.relu(self.conv2(x))
        # print(x.shape)
        x = self.pool(x)
        # print(x.shape)

        # print("third convolution")
        x = F.relu(self.conv3(x))
        # print(x.shape)
        x = self.pool3(x)
        # print(x.shape)

        x = x.view(-1, 64 * 2 * 150)
        # print(x.shape)
        # Computes the activation of the first fully connected layer
        # Size changes from (1, 96000) to (1, 256)
        x = F.relu(self.fc1(x))
        # print(x.shape)
        
        # Size changes from (1, 256) to (1, 128)
        x = F.relu(self.fc2(x))
        # print(x.shape)
        # Computes the second fully connected layer (activation applied later)
        # Size changes from (1, 128) to (1, 64)
        x = F.relu(self.fc3(x))
        
        # 1,64 -> 1,10
        x = self.fc4(x)
        # print(x.shape)
        # print('next')
        # print(x.shape)
        return(x)

categories = {
    "blues": 0,
    "classical": 1,
    "country": 2,
    "disco": 3,
    "hiphop": 4,
    "jazz": 5,
    "metal": 6,
    "pop": 7,
    "reggae": 8,
    "rock": 9
}

# reshape a 2D array if the size does not match
def reshape_array(arr, expected_row_num, expected_col_num):
    if arr.shape == (expected_row_num, expected_col_num):
        return arr

    result = arr
    # remove extra column for larger dimensions
    if arr.shape[1] > expected_col_num:
        result = result[:, :expected_col_num]
    if arr.shape[0] > expected_row_num:
        result = result[:expected_row_num, :]

    # for smaller dimensions, pad with zeroes:
    zeros_arr = np.zeros((expected_row_num, expected_col_num))
    zeros_arr[:result.shape[0],:result.shape[1]] = result
    result = zeros_arr

    return result

def createLossAndOptimizer(net, learning_rate=0.001):
    #Loss function
    loss = torch.nn.CrossEntropyLoss()
    
    #Optimizer
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    return (loss, optimizer)

def trainNet(net, train_loader, val_loader, n_epochs, batch_size, learning_rate):
    
    #Print all of the hyperparameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    print("epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    
    # Create our loss and optimizer functions
    loss, optimizer = createLossAndOptimizer(net, learning_rate)
    
    # Time for printing
    training_start_time = time.time()
    n_batches = len(train_loader)
    # Loop for n_epochs
    for epoch in range(n_epochs):
        
        running_loss = 0.0
        print_every = 1
        start_time = time.time()
        total_train_loss = 0
        
        for i, data in enumerate(train_loader, 0):

            # Get inputs
            inputs, labels = data
            # print(labels)
            # print(labels)
            # Set the parameter gradients to zero
            optimizer.zero_grad()
            
            # Forward pass, backward pass, optimize
            outputs = net(inputs)
    
            loss_size = loss(outputs, labels.long())
            loss_size.backward()
            optimizer.step()
            
            #Print statistics
            running_loss += loss_size.item()
            total_train_loss += loss_size.item()
            
            # Print every 10th batch of an epoch
            if (i + 1) % (print_every + 1) == 0:
                print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
                        epoch+1, int(100 * (i+1) / n_batches), running_loss / print_every, time.time() - start_time))
                # Reset running loss and time
                running_loss = 0.0
                start_time = time.time()
        train_cost.append(total_train_loss / len(train_loader))
            
        # At the end of the epoch, do a pass on the validation set
        total_val_loss = 0
        for inputs, labels in val_loader:
            # Forward pass
            # print(labels)
            val_outputs = net(inputs)
            # print(val_outputs)
            val_loss_size = loss(val_outputs, labels.long())
            total_val_loss += val_loss_size.item()
        print("Validation loss = {:.2f}".format(total_val_loss / len(val_loader)))
        val_cost.append(total_val_loss / len(val_loader))
        epoch_num.append(epoch + 1)
    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))


dataset = []
path = "../data_audio/genres_original"
train_cost = []
val_cost = []
epoch_num = []

for label in os.listdir(path):
    song_folder_path = os.path.join(path, label)
    songs = os.listdir(song_folder_path) # All of the songs are here
    
    # Convert song to .wav format if it is not in .wav format. Otherwise skip
    for song in songs:
        song_file_path = os.path.join(song_folder_path, song)
        
        # convert to mel spectrogram
        y, sr = librosa.load(song_file_path)
        num_of_mfccs = 20
        mfcc = reshape_array(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=num_of_mfccs, n_fft=2048, hop_length=1024), 20, 1200)
        
        # assign category
        category = song.split(".")[0]
        category_index = categories[category]

        dataset.append((mfcc, category_index))
        # print("song {} is done!".format(song))
    # break

# randomly shuffle the dataset
# print("breakpoint 1")
batch_size = 40
n_training_samples = 800
n_val_samples = 99
torch.manual_seed(13)
np.random.seed(13)
train_sampler = SubsetRandomSampler(np.arange(n_training_samples, dtype=np.int64))
val_sampler = SubsetRandomSampler(np.arange(n_val_samples, dtype=np.int64))

train_mfccs = []
train_labels = []
test_mfccs = []
test_labels = []
val_mfccs = []
val_labels = []

for i in range(len(dataset)):
    j  = i % 100
    x, label = dataset[i]
    if j < 80:
        train_mfccs.append(x)
        train_labels.append(label)
    elif j < 90:
        test_mfccs.append(x)
        test_labels.append(label)
    else:
        val_mfccs.append(x)
        val_labels.append(label)

train_mfccs = np.array(train_mfccs)
train_labels = np.array(train_labels)
test_mfccs = np.array(test_mfccs)
test_labels = np.array(test_labels)
val_mfccs = np.array(val_mfccs)
val_labels = np.array(val_labels)

# print(train_mfccs.shape[0])
# print(test_mfccs.shape[0])
# print(val_mfccs.shape[0])

'''train_mfccs = np.array(all_features[:800])
test_mfccs = np.array(all_features[800:900])
val_mfccs = np.array(all_features[900:])

# labels
train_labels = np.array(all_labels[:800])
test_labels = np.array(all_labels[800:900])
val_labels = np.array(all_labels[900:])'''

mean_var_dict = {}

for mfcc_2d in train_mfccs:
    for i in range(mfcc_2d.shape[0]):
        row = mfcc_2d[i]
        if i not in mean_var_dict:
            mean_var_dict[i] = { 'entries': [] }
        mean_var_dict[i]['entries'].append(row)

# Compute mean and variance for each row in train set
for mfcc_index, d in mean_var_dict.copy().items():
    flattened_entries = np.array(d['entries']).flatten()
    mean = np.mean(flattened_entries)
    std = np.std(flattened_entries)
    mean_var_dict[mfcc_index]['mean'] = mean
    mean_var_dict[mfcc_index]['std'] = std

# normalize every mfcc in train
normalized_train_mfcc = []
for mfcc_2d in train_mfccs:
    norm_mfcc_2d = []
    for i in range(mfcc_2d.shape[0]):
        normalized_row = (mfcc_2d[i] - mean_var_dict[i]['mean']) / mean_var_dict[i]['std']
        norm_mfcc_2d.append(normalized_row)
    normalized_train_mfcc.append([norm_mfcc_2d])

normalized_train_mfcc = np.array(normalized_train_mfcc)
# print(normalized_train_mfcc)
train_dataset = TensorDataset(torch.Tensor(normalized_train_mfcc), torch.Tensor(train_labels))


# Reset for test
mean_var_dict = {}

for mfcc_2d in test_mfccs:
    for i in range(mfcc_2d.shape[0]):
        row = mfcc_2d[i]
        if i not in mean_var_dict:
            mean_var_dict[i] = { 'entries': [] }
        mean_var_dict[i]['entries'].append(row)

# Compute mean and variance for each row in test set
for mfcc_index, d in mean_var_dict.copy().items():
    flattened_entries = np.array(d['entries']).flatten()
    mean = np.mean(flattened_entries)
    std = np.std(flattened_entries)
    mean_var_dict[mfcc_index]['mean'] = mean
    mean_var_dict[mfcc_index]['std'] = std

# normalize every mfcc in test
normalized_test_mfcc = []
for mfcc_2d in test_mfccs:
    norm_mfcc_2d = []
    for i in range(mfcc_2d.shape[0]):
        normalized_row = (mfcc_2d[i] - mean_var_dict[i]['mean']) / mean_var_dict[i]['std']
        norm_mfcc_2d.append(normalized_row)
    normalized_test_mfcc.append([norm_mfcc_2d])

normalized_test_mfcc = np.array(normalized_test_mfcc)
test_dataset = TensorDataset(torch.Tensor(normalized_test_mfcc), torch.Tensor(test_labels))


# Reset for validation
mean_var_dict = {}
for mfcc_2d in val_mfccs:
    for i in range(mfcc_2d.shape[0]):
        row = mfcc_2d[i]
        if i not in mean_var_dict:
            mean_var_dict[i] = { 'entries': [] }
        mean_var_dict[i]['entries'].append(row)

# Compute mean and variance for each row in val set
for mfcc_index, d in mean_var_dict.copy().items():
    flattened_entries = np.array(d['entries']).flatten()
    mean = np.mean(flattened_entries)
    std = np.std(flattened_entries)
    mean_var_dict[mfcc_index]['mean'] = mean
    mean_var_dict[mfcc_index]['std'] = std

# normalize every mfcc in val
normalized_val_mfcc = []
for mfcc_2d in val_mfccs:
    norm_mfcc_2d = []
    for i in range(mfcc_2d.shape[0]):
        normalized_row = (mfcc_2d[i] - mean_var_dict[i]['mean']) / mean_var_dict[i]['std']
        norm_mfcc_2d.append(normalized_row)
    normalized_val_mfcc.append([norm_mfcc_2d])

normalized_val_mfcc = np.array(normalized_val_mfcc)

val_dataset = TensorDataset(torch.Tensor(normalized_val_mfcc), torch.Tensor(val_labels))

train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)

cnn = CNN()
trainNet(cnn, train_loader=train_loader, val_loader=val_loader, batch_size=batch_size, n_epochs=25, learning_rate=0.001)

correct = 0
total = 0
counter = 1
for data in test_loader:
    mfcc, labels = data
    outputs = cnn(mfcc)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
print('Accuracy of the network on the test mfccs: %f %%' % (100 * float(correct) / total))

correct = 0
total = 0
counter = 1
for data in train_loader:
    mfcc, labels = data
    outputs = cnn(mfcc)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
print ('Accuracy of network on train mfccs: %f %%' % (100 * float(correct) / total))

fig = plt.figure()
plt.plot(epoch_num, train_cost, label='Train Cost')
plt.plot(epoch_num, val_cost, label='Validation Cost')
plt.ylabel('Cost')
plt.xlabel('Epoch')
plt.legend()
plt.show()
