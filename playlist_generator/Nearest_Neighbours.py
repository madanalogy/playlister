import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import neighbors
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

def filter(data, column, value):
    return data.loc[[(value in row) for row in data[column]]]

def display_songs(songs):
    main_song = songs[0]
    message = f'Song: {main_song["name"]} by {main_song.artists}\n'
    message += 'Neighbors:\n'
    list_number = 1

    # Only one song, so first list of indices
    for song in songs[1:]:
        message += f'{list_number:2d}. {song["name"]} by {song.artists}\n'
        list_number += 1

    print(message)

def find_songs_by_genre(song, n=10):
    song_genre = song.genres[0] # first genre
    song_features = song[features]

    song_datas = data.append(song)
    x = song_datas[features]
    x_scaled = StandardScaler().fit_transform(x)
    song_scaled = x_scaled[len(x_scaled) - 1]

    nbrs = NearestNeighbors(n_neighbors = n + 1, algorithm = 'ball_tree').fit(x_scaled)
    distances, indices = nbrs.kneighbors(np.array([song_scaled]))

    songs = []
    for song_id in indices[0]:
        songs.append(song_datas.iloc[song_id])

    return songs

def find_songs_by_genre_PCA(song, n=10, components=6):
    song_genre = song.genres[0] # first genre
    song_features = song[features]

    song_datas = data.append(song)
    x = song_datas[features]
    x_scaled = StandardScaler().fit_transform(x)

    pca = PCA(n_components=components)
    principalComponents = pca.fit_transform(x_scaled)
    song_scaled = principalComponents[len(principalComponents) - 1]

    nbrs = NearestNeighbors(n_neighbors = n + 1, algorithm = 'ball_tree').fit(principalComponents)
    distances, indices = nbrs.kneighbors(np.array([song_scaled]))

    songs = []
    for song_id in indices[0]:
        songs.append(song_datas.iloc[song_id])

    return songs

def find_songs_by_valence(song, n=10):
    song_genre = song.genres[0] # first genre
    song_features = song[features]

    song_datas = data.append(song)
    x = song_datas[features]
    x_scaled = StandardScaler().fit_transform(x)
    song_scaled = x_scaled[len(x_scaled) - 1]

    nbrs = NearestNeighbors(n_neighbors = n + 1, algorithm = 'ball_tree').fit(x_scaled)
    distances, indices = nbrs.kneighbors(np.array([song_scaled]))

    songs = []
    for song_id in indices[0]:
        songs.append(song_datas.iloc[song_id])

    return songs


major_genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
features = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

# TODO input query genre
data = pd.read_csv('popular_pop_songs.csv')

# TODO input song needs to be based on name and artist
my_songs = find_songs_by_genre(data.iloc[0]) 
my_songs_PCA = find_songs_by_genre_PCA(data.iloc[0])

display_songs(my_songs)
display_songs(my_songs_PCA)