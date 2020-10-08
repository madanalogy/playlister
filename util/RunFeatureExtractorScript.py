import librosa
import numpy as np
import math
import os, sys, getopt
import csv

from FileConversion import FileConversion

dataset_root_directory = None

def usage():
    print("Please specify the directory of the genres_original dataset. E.g. ../data/genres_original")
    print("This script will assume that the name of the folder is the label of all the songs inside it.")

try:
    dataset_root_directory = os.path.abspath(sys.argv[1])
except IndexError:
    usage()
    sys.exit(2)

if dataset_root_directory == None:
    sys.exit(2)

song_label_folders = os.listdir(dataset_root_directory)
csv_rows = []

for label in song_label_folders:
    song_folder_path = os.path.join(dataset_root_directory, label)
    songs = os.listdir(song_folder_path) # All of the songs are here
    
    # Convert song to .wav format if it is not in .wav format. Otherwise skip
    for song in songs:
        song_file_path = os.path.join(song_folder_path, song)
        formatted_song_file_path = FileConversion(song_file_path).convert()

        if formatted_song_file_path == None:
            continue
        
        audio_features = AudioFeatureExtractor(formatted_song_file_path, label).extract_features()
        csv_rows.append(audio_features)
        print("song {} is done!".format(song))

if len(csv_rows) == 0:
    print("No rows in data")
    sys.exit(2)

with open("out.csv", 'w', newline='') as csvfile:
    headers = [key for key, value in csv_rows[0].items()]
    writer = csv.DictWriter(csvfile, fieldnames=headers)

    writer.writeheader()
    sorted_rows = sorted(csv_rows, key=lambda x: x['filename'])
    writer.writerows(sorted_rows)