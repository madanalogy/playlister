import sys, getopt, os
import csv

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.decomposition import PCA

def usage():
    print("Please specify csv data file path. E.g python svm.py ../data/data.csv")

data_file_name = None
try:
    data_file_name = os.path.abspath(sys.argv[1])
except IndexError:
    usage()
    sys.exit(2)

if data_file_name == None:
    sys.exit(2)

data = pd.read_csv(data_file_name)
# features = ['acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness', 'liveness','loudness','speechiness','tempo','valence','popularity','key','mode','count']
features = ['chroma_stft_mean', 'chroma_stft_var', 'rms_mean' , 'rms_var', 'spectral_centroid_mean', 'spectral_centroid_var', 'spectral_bandwidth_mean', 'spectral_bandwidth_var', 'rolloff_mean', 'rolloff_var', 'zero_crossing_rate_mean', 'zero_crossing_rate_var' , 'harmony_mean', 'harmony_var', 'percussive_mean' ,'percussive_var', 
'tempo', 'mfcc1_mean', 'mfcc1_var', 'mfcc2_mean', 'mfcc2_var' , 'mfcc3_mean', 'mfcc3_var', 'mfcc4_mean' , 'mfcc4_var', 'mfcc5_mean' , 'mfcc5_var' , 'mfcc6_mean', 'mfcc6_var', 'mfcc7_mean' , 'mfcc7_var', 'mfcc8_mean', 'mfcc8_var', 'mfcc9_mean', 'mfcc9_var', 'mfcc10_mean', 'mfcc10_var',
'mfcc11_mean', 'mfcc11_var', 'mfcc12_mean', 'mfcc12_var', 'mfcc13_mean', 'mfcc13_var', 'mfcc14_mean', 'mfcc14_var', 'mfcc15_mean', 'mfcc15_var', 'mfcc16_mean', 'mfcc16_var', 'mfcc17_mean', 'mfcc17_var', 'mfcc18_mean', 'mfcc18_var', 'mfcc19_mean', 'mfcc19_var', 'mfcc20_mean', 'mfcc20_var', 'spectral_flatness_mean', 'spectral_flatness_var']
df_X = data[features].copy()
df_y = data["label"].copy()

X_train, X_test, y_train, y_test = train_test_split(
    df_X, df_y, test_size=0.2, random_state=0, stratify=df_y
)

# PCA
'''pca = PCA(n_components=14, svd_solver='randomized',
          whiten=True).fit(X_train)

X_train = pca.transform(X_train)
X_test = pca.transform(X_test)'''


print("Train set size: {}, Test set size: {}".format(len(X_train), len(X_test)))
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

classifier = SGDClassifier(max_iter=100000, tol=1e-3, loss='log')
classifier.fit(X_train_scaled, y_train.values.ravel())

print('score using training data: {}'.format(classifier.score(X_train_scaled, y_train.values.ravel())))
print('score using unseen datapoints: {}'.format(classifier.score(X_test_scaled, y_test.values.ravel())))

# About 8%