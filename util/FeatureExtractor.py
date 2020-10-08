import librosa
import numpy as np
import math
import os, sys, getopt
import csv

class AudioFeatureExtractor(object):
    def __init__(self, audio_file_path, label):
        self.audio_file_path = audio_file_path
        self.label = label

    def extract_features(self):
        result_dict = {}
        y, sr = librosa.load(self.audio_file_path)

        # Zeroth field - audio file path
        result_dict["filename"] = os.path.basename(self.audio_file_path)

        # First field - song length
        length = y.shape[0]
        result_dict["length"] = length

        # Second field - chroma frequency mean and variance
        chromagram = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_stft_mean = np.mean(np.mean(chromagram, axis=1)) # Calculate the mean for each chroma bin across time frame, then average it across all chroma bins
        chroma_stft_var = np.var(chromagram.flatten())
        # print("chroma mean: {} \nchroma variance: {}".format(chroma_stft_mean, chroma_stft_var))
        result_dict["chroma_stft_mean"] = chroma_stft_mean
        result_dict["chroma_stft_var"] = chroma_stft_var

        # Third field - rms mean and variance
        rms = librosa.feature.rms(y=y)
        rms_mean = np.mean(rms)
        rms_var = np.var(rms)
        # print("rms mean: {} \nrms variance: {}".format(rms_mean, rms_var))
        result_dict["rms_mean"] = rms_mean
        result_dict["rms_var"] = rms_var

        # Fourth field - spectral centroid mean and variance
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_centroid_mean = np.mean(centroid)
        spectral_centroid_var = np.var(centroid)
        # print("spectral centroid mean: {}\nspectral centroid variance: {}".format(spectral_centroid_mean, spectral_centroid_var))
        result_dict["spectral_centroid_mean"] = spectral_centroid_mean
        result_dict["spectral_centroid_var"] = spectral_centroid_var


        # Fifth field - spectral bandwidth mean and variance
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spectral_bandwidth_mean = np.mean(bandwidth)
        spectral_bandwidth_var = np.var(bandwidth)
        # print("spectral bandwidth mean: {}\nspectral bandwidth variance: {}".format(spectral_bandwidth_mean, spectral_bandwidth_var))
        result_dict["spectral_bandwidth_mean"]= spectral_bandwidth_mean
        result_dict["spectral_bandwidth_var"] = spectral_bandwidth_var


        # Sixth field - spectral rolloff mean and variance
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        rolloff_mean = np.mean(rolloff)
        rolloff_var = np.var(rolloff)
        # print("spectral rolloff mean: {}\nspectral rolloff variance: {}".format(rolloff_mean, rolloff_var))
        result_dict["rolloff_mean"] = rolloff_mean
        result_dict["rolloff_var"] = rolloff_var


        # Seventh field - zero crossing rate mean and variance
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)
        zero_crossing_rate_mean = np.mean(zero_crossing_rate)
        zero_crossing_rate_var = np.var(zero_crossing_rate)
        # print("zero crossing rate mean: {}\nzero crossing rate variance: {}".format(zero_crossing_rate_mean, zero_crossing_rate_var))
        result_dict["zero_crossing_rate_mean"]= zero_crossing_rate_mean
        result_dict["zero_crossing_rate_var"] = zero_crossing_rate_var

        # Eighth field - harmonic mean and variance
        harmony = librosa.effects.harmonic(y=y)
        harmony_mean = np.mean(harmony)
        harmony_var = np.var(harmony)
        # print("harmonic mean: {}\nharmonic variance: {}".format(harmony_mean, harmony_var))
        result_dict["harmony_mean"] = harmony_mean
        result_dict["harmony_var"] = harmony_var

        # Ninth field - percussive mean and variance
        percussive = librosa.effects.percussive(y=y)
        percussive_mean = np.mean(percussive)
        percussive_var = np.var(percussive)
        # print("percussive mean: {}\npercussive variance: {}".format(percussive_mean, percussive_var))
        result_dict["percussive_mean"] = percussive_mean
        result_dict["percussive_var"] = percussive_var

        # Tenth field - tempo
        hop_length = 512
        oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr, hop_length=hop_length)
        tempo = librosa.beat.tempo(onset_envelope=oenv, sr=sr, hop_length=hop_length)[0]
        # print("tempo: {}".format(tempo))
        result_dict["tempo"] = tempo

        # Eleventh field - 20 different mfccs
        num_of_mfccs = 20
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=num_of_mfccs)
        # Means are calculated across all time frame for every mfcc
        for i in range(20):
            mfcc_mean = np.mean(mfcc[i])
            mfcc_var = np.var(mfcc[i])
            # print("mfcc{} mean: {}, var: {}".format(i+1, mfcc_mean, mfcc_var))
            result_dict["mfcc{}_mean".format(i+1)] = mfcc_mean
            result_dict["mfcc{}_var".format(i+1)] = mfcc_var

        # Twelfth field - spectral flatness (THIS ISNT FOUND IN THE FEATURES.csv file!) - One additional feature to play around
        flatness = librosa.feature.spectral_flatness(y=y)
        spectral_flatness_mean = np.mean(flatness)
        spectral_flatness_var = np.var(flatness)
        # print("spectral flatness mean :{}\nspectral flatness variance: {}".format(spectral_flatness_mean, spectral_flatness_var))
        result_dict["spectral_flatness_mean"] = spectral_flatness_mean
        result_dict["spectral_flatness_var"] = spectral_flatness_var

        # Include the label as well
        result_dict["label"] = self.label

        return result_dict

'''Usage
audio_file_path = '../data/genres_original/blues/blues.00000.wav'
audio_features = AudioFeatureExtractor(audio_file_path).extract_features()
print(audio_features)'''
