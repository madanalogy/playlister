import numpy as np
import pandas as pd

import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split


# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop('Genre')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds

# # Input builders
# def input_fn_train:
#   # Returns tf.data.Dataset of (x, y) tuple where y represents label's class
#   # index.
#   pass
# def input_fn_eval:
#   # Returns tf.data.Dataset of (x, y) tuple where y represents label's class
#   # index.
#   pass
# def input_fn_predict:
#   # Returns tf.data.Dataset of (x, None) tuple.
#   pass


DATA_FILE_PATH = "../data/data_clean.csv"
SELECT_COLUMNS = ['Danceability', 'Energy', 'Key', 'Loudness', 'Mode', 'Speechness', 'Acousticness', 'Instrumentalness',
                  'Liveness', 'Valence', 'Tempo', 'Genre']

data = pd.read_csv(DATA_FILE_PATH)[SELECT_COLUMNS]
label_vocab = data['Genre'].unique().tolist()

train, test = train_test_split(data, test_size=0.2)
print(len(train), 'train examples')
print(len(test), 'test examples')
print(len(label_vocab), 'classes')

# train_ds = df_to_dataset(train)
# test_ds = df_to_dataset(test, shuffle=False)

feature_columns = []
for feature in SELECT_COLUMNS:
    if feature != 'Genre':
        feature_columns.append(feature_column.numeric_column(feature))

feature_layer = layers.DenseFeatures(feature_columns)

classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[128, 128],
    n_classes=len(label_vocab),
    label_vocabulary=label_vocab)

classifier.train(input_fn=lambda: df_to_dataset(train, batch_size=5))
metrics = classifier.evaluate(input_fn=lambda: df_to_dataset(test, shuffle=False, batch_size=5))
# predictions = classifier.predict(input_fn=input_fn_predict)

print(metrics)
