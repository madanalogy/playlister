import os
import sys

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import plot_confusion_matrix

from util.FeatureExtractor import AudioFeatureExtractor

data_file_name = os.path.dirname(os.path.abspath(__file__)) + "/features.csv"
data = pd.read_csv(data_file_name)

features = ['chroma_stft_mean', 'chroma_stft_var', 'rms_mean', 'rms_var', 'spectral_centroid_mean',
            'spectral_centroid_var', 'spectral_bandwidth_mean', 'spectral_bandwidth_var', 'rolloff_mean', 'rolloff_var',
            'zero_crossing_rate_mean', 'zero_crossing_rate_var', 'harmony_mean', 'harmony_var', 'percussive_mean',
            'percussive_var', 'tempo',
            'mfcc1_mean', 'mfcc1_var', 'mfcc2_mean', 'mfcc2_var', 'mfcc3_mean', 'mfcc3_var', 'mfcc4_mean', 'mfcc4_var',
            'mfcc5_mean', 'mfcc5_var', 'mfcc6_mean', 'mfcc6_var', 'mfcc7_mean', 'mfcc7_var', 'mfcc8_mean', 'mfcc8_var',
            'mfcc9_mean', 'mfcc9_var', 'mfcc10_mean', 'mfcc10_var', 'mfcc11_mean', 'mfcc11_var', 'mfcc12_mean',
            'mfcc12_var', 'mfcc13_mean', 'mfcc13_var', 'mfcc14_mean', 'mfcc14_var', 'mfcc15_mean', 'mfcc15_var',
            'mfcc16_mean', 'mfcc16_var', 'mfcc17_mean', 'mfcc17_var', 'mfcc18_mean', 'mfcc18_var', 'mfcc19_mean',
            'mfcc19_var', 'mfcc20_mean', 'mfcc20_var', 'spectral_flatness_mean', 'spectral_flatness_var']

df_X = data[features].copy()
df_y = data["label"].copy()

X_train, X_test, y_train, y_test = train_test_split(
    df_X, df_y, test_size=0.2, random_state=0, stratify=df_y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# print("Genre: Beginning training...\nTrain set size: {}, Test set size: {}".format(len(X_train), len(X_test)))

classifier = MLPClassifier(solver='adam', alpha=1e-5, random_state=0, hidden_layer_sizes=(300, 50, 20), max_iter=200)
# 300, 50, 20 - 0.8

classifier.fit(X_train_scaled, y_train.values.ravel())
# print("Genre: Model has been trained!")


# Takes in a list of audio paths.
# Outputs label based on MLPClassifier
def predict(audio_file_paths):
    X = [AudioFeatureExtractor(fp, None).extract_features() for fp in audio_file_paths]
    pred = pd.DataFrame.from_dict(data=X)[features]
    return classifier.predict(scaler.transform(pred))


def get_score():
    return classifier.score(X_test_scaled, y_test), classifier.score(X_train_scaled, y_train)


def plot_cnf_mat():
    title = "Confusion Matrix"
    disp = plot_confusion_matrix(classifier, X_test_scaled, y_test,
                                 cmap=plt.cm.Blues)
    disp.ax_.set_title(title)
    plt.show()


if __name__ == '__main__':
    # Simple usage of application
    try:
        data_file_name = os.path.abspath(sys.argv[1])
    except IndexError:
        print("Genre: Using default features.csv")

    print("Genre: Reading data from {}".format(data_file_name))
    if data_file_name is None:
        sys.exit(2)
    
    '''
    test_inputs = ['../data_audio/genres_original/blues/blues.00010.wav',
                   '../data_audio/genres_original/blues/blues.00001.wav']
    print(predict(test_inputs))  # ['blues', 'blues']
    '''
