import numpy as np
import pandas as pd

import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split


# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle=True, batch_size=16):
    dataframe = dataframe.copy()
    labels = dataframe.pop('Genre')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds


DATA_FILE_PATH = "../data/data_clean.csv"
SELECT_COLUMNS = ['Danceability', 'Energy', 'Key', 'Loudness', 'Mode', 'Speechness', 'Acousticness', 'Instrumentalness',
                  'Liveness', 'Valence', 'Tempo', 'Genre']

data = pd.read_csv(DATA_FILE_PATH)[SELECT_COLUMNS]
label_vocab = data['Genre'].unique().tolist()

train, test = train_test_split(data, test_size=0.2)
print(len(train), 'train examples')
print(len(test), 'test examples')
print(len(label_vocab), 'classes')

# Normalise data
train_g = train.pop('Genre')
train = (train - train.mean()) / train.std()
train['Genre'] = train_g

test_g = test.pop('Genre')
test = (test - test.mean()) / test.std()
test['Genre'] = test_g

# Set up feature columns
feature_columns = []
for feature in SELECT_COLUMNS:
    if feature != 'Genre':
        feature_columns.append(feature_column.numeric_column(feature))

feature_layer = layers.DenseFeatures(feature_columns)

classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[16, 20, 24, 24],
    n_classes=len(label_vocab),
    label_vocabulary=label_vocab)

classifier.train(input_fn=lambda: df_to_dataset(train))
eval_result = classifier.evaluate(input_fn=lambda: df_to_dataset(test, shuffle=False))
# predictions = classifier.predict(input_fn=input_fn_predict)

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
