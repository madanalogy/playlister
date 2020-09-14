import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import functools
from sklearn.model_selection import train_test_split

'''
Note: just experimenting with neural networks and the model is not carefully set up.
right now it only has a test accuracy of slight above 13%.
'''

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

DATA_FILE_PATH = "../data/data_clean.csv"
SELECT_COLUMNS = ['Danceability', 'Energy', 'Key', 'Loudness', 'Mode', 'Speechness', 'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo', 'Genre']

full_dataframe = pd.read_csv(DATA_FILE_PATH)[SELECT_COLUMNS]

# map Genres to integers
genre_set = set()
for genre in full_dataframe["Genre"]:
    genre_set.add(genre.lower())

genre_list = list(genre_set)
genre_int_column = []
print("mapping genres to ints...")
for index in range(len(full_dataframe["Genre"])):
    genre_string = full_dataframe["Genre"][index]
    genre_int_column.append(genre_list.index(genre_string))
full_dataframe['Genre_label'] = genre_int_column

# Then remove the original Genre column
full_dataframe.drop("Genre", axis=1, inplace=True)

# Separate the predictor and response data.
df_X = full_dataframe[full_dataframe.columns[full_dataframe.columns != "Genre_label"]].copy()
df_y = full_dataframe["Genre_label"].copy()

# (random_state): we use a fixed random seed so we get the
# same results every time we run the code.
X_train, X_test, y_train, y_test = train_test_split(
    df_X, df_y, test_size=0.2, random_state=0
)

print(
    "Number of training instances: ",
    len(X_train),
    "\nNumber of test instances: ",
    len(X_test),
)

# change dataframe to numpy array
X_train_array = X_train.to_numpy()
y_train_array = y_train.to_numpy()
X_test_array = X_test.to_numpy()
y_test_array = y_test.to_numpy()

# Normalise training data
desc = X_train.describe()
MEAN = np.array(desc.T['mean'])
STD = np.array(desc.T['std'])
for row in X_train_array:
    for index in range(len(row)):
        row[index] = row[index] - MEAN[index] / STD[index]

# calculate number of output nodes
number_of_genres = len(full_dataframe["Genre_label"].value_counts())
print("number of genres is: ", number_of_genres)

print("now our data looks like this: ", X_train_array[:10])

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(number_of_genres),
])

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    # optimizer='adam',
    metrics=['accuracy'])

model.fit(X_train_array, y_train_array, epochs=100)

# Normalise test data
desc_test = X_test.describe()
MEAN_TEST = np.array(desc_test.T['mean'])
STD_TEST = np.array(desc_test.T['std'])
print("MEAN_TEST: ", MEAN_TEST)
print("STD_TEST: ", STD_TEST)

for row in X_test_array:
    for index in range(len(row)):
        row[index] = row[index] - MEAN_TEST[index] / STD_TEST[index]

# evaluate model accuracy
test_loss, test_acc = model.evaluate(X_test_array,  y_test_array, verbose=1)

print('\nTest accuracy:', test_acc)
