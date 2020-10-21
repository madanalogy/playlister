import os

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

# from unidecode import unidecode

major_genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

features = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']


data_dir = os.path.dirname(os.path.abspath(__file__))


def get_file_path(file):
    return f'{data_dir}/{file}'


data = pd.read_csv(get_file_path('songs_with_genres.csv'))
spotify_song_names = data['simple_name'].sort_values()


def find_spotify_info(song_number):
    return data.iloc[song_number]


def find_song_numbers_by_keyword(keyword):
    song_numbers = []
    for i, song_name in spotify_song_names.iteritems():
        if keyword in song_name:
            song_numbers.append(i)
    return song_numbers


def display_songs(songs):
    message = f'Your Playlist:\n'
    # Number playlist songs from 1 onwards
    list_number = 1
    for _, song in songs.iterrows():
        message += f'{list_number}. {song["name"]} [by {song.artists}]\n'
        list_number += 1
    print(message)


def find_songs_by_features(seeds, n=10, pca=True, components=7):
    x = data[features]
    x_scaled = StandardScaler().fit_transform(x)

    # TODO: Catch ID not in song list, retrieve features from Spotify

    if pca:
        pca = PCA(n_components=components)
        principal_components = pca.fit_transform(x_scaled)
        rows_with_seeds = principal_components[seeds]
        neighbors = NearestNeighbors(n_neighbors=n + 1, algorithm='ball_tree').fit(principal_components)
    else:
        rows_with_seeds = x_scaled[seeds]
        neighbors = NearestNeighbors(n_neighbors=n + 1, algorithm='ball_tree').fit(x_scaled)

    aggregate_features = np.array([np.array(rows_with_seeds).mean(axis=0)])
    distances, indices = neighbors.kneighbors(aggregate_features)

    songs = data.iloc[indices[0]]
    if len(seeds) <= 1:
        songs = songs[1:]
    else:
        songs = songs[:-1]
    return songs


def find_songs_by_valence(genre, valence, n=10):
    data = pd.read_csv(get_file_path(f'popular_{genre}_songs.csv'))
    diff = []
    for i, song in data.iterrows():
        diff.append(abs(song['valence'] - valence))

    data['diff'] = diff

    data = data.sort_values(['diff', 'popularity'], ascending=[True, False])
    songs = data.head(n)
    return songs
