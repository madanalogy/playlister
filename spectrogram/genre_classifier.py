import os
import glob
import random
import librosa
import librosa.display
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

path = "../data/audio_data/genres_original/"

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

# array of tuples of (spectrogram, label)
dataset = []

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
        
# iterate through all audio files
for root, dirs, files in os.walk(path):
   for file_name in files:
        if file_name == ".DS_Store":
            continue

        # TODO: figure out why this file doesn't work
        if file_name == "jazz.00054.wav":
            file_name = "jazz.00055.wav"

        full_path = os.path.join(root, file_name)
        
        # convert to mel spectrogram
        y, sr = librosa.load(full_path)
        mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024)
        mel_spect = reshape_array(mel_spect, 128, 647)
        
        # assign category
        category = file_name.split(".")[0]
        category_index = categories[category]

        dataset.append((mel_spect, category_index))

# randomly shuffle the dataset
random.seed(20)
random.shuffle(dataset)

# assign train and test sets
all_features = list(map(lambda x: x[0], dataset))
all_labels = list(map(lambda x: x[1], dataset))

train_spectrograms = np.array(all_features[:800])
train_labels = np.array(all_labels[:800])

test_spectrograms = np.array(all_features[800:])
test_labels = np.array(all_labels[800:])

print("shape of spectrograms is", train_spectrograms.shape)
print("shape of labels is", train_labels.shape)

print("shape of test spectrograms is", test_spectrograms.shape)
print("shape of test labels is", test_labels.shape)

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(128, 647)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10),
])

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['accuracy'])

model.fit(train_spectrograms, train_labels, epochs=5)

# test model
test_loss, test_acc = model.evaluate(test_spectrograms, test_labels, verbose=2)

print('\nTest accuracy:', test_acc)
