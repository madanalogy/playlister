import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import neighbors
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

major_genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

features = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

def filter(data, column, value):
    return data.loc[[(value in row) for row in data[column]]]

def display_songs_1(songs, genre, valence):
    message = f'Genre: {genre}\n'
    message += f'Valence: {valence}\n'
    message += f'Songs:\n'
    list_number = 1

    # Only one song, so first list of indices
    for i, song in songs.iterrows():
        message += f'{list_number:2d}. {song["name"]} by {song.artists}\n'
        list_number += 1

    print(message)

def display_songs_2(songs):
    main_song = songs[0]
    message = f'Song: {main_song["name"]} by {main_song.artists}\n'
    message += 'Neighbors:\n'
    list_number = 1

    # Only one song, so first list of indices
    for song in songs[1:]:
        message += f'{list_number:2d}. {song["name"]} by {song.artists}\n'
        list_number += 1

    print(message)

def find_songs_by_genre(song_index, n=10):
    x = data[features]
    x_scaled = StandardScaler().fit_transform(x)
    song_scaled = x_scaled[song_index]

    nbrs = NearestNeighbors(n_neighbors = n + 1, algorithm = 'ball_tree').fit(x_scaled)
    distances, indices = nbrs.kneighbors(np.array([song_scaled]))

    songs = []
    for song_id in indices[0]:
        songs.append(data.iloc[song_id])

    return songs

def find_songs_by_genre_PCA(song_index, n=10, components=7):
    x = data[features]
    x_scaled = StandardScaler().fit_transform(x)

    pca = PCA(n_components=components)
    principalComponents = pca.fit_transform(x_scaled)
    song_scaled = principalComponents[song_index]

    nbrs = NearestNeighbors(n_neighbors = n + 1, algorithm = 'ball_tree').fit(principalComponents)
    distances, indices = nbrs.kneighbors(np.array([song_scaled]))

    songs = []
    for song_id in indices[0]:
        songs.append(data.iloc[song_id])

    return songs

def find_songs_by_valence(genre, valence, n=10):
    data = pd.read_csv('popular_' + genre + '_songs.csv')
    diff = []
    for i, song in data.iterrows():
        diff.append(abs(song['valence'] - valence))

    data['diff'] = diff

    data = data.sort_values(['diff', 'popularity'], ascending = [True, False])
    songs = data.head(n)
    return songs

workflow = int(input ("Choose a workflow: "))
playlist_length = int(input ("Enter the length of the playlist (must be less than 3000): "))

if workflow == 1:
    # Workflow 1
    genre = input ("Input a genre: ")
    valence = float(input ("Input a valence: "))

    songs = find_songs_by_valence(genre, valence, n=playlist_length)
    display_songs_1(songs, genre, valence)

elif workflow == 2:
    # Workflow 2
    song_id = input ("Input a Spotify ID: ")

    data = pd.read_csv('songs_with_genres.csv')
    song_index = data.loc[data['id'] == song_id].index[0]

    my_songs = find_songs_by_genre(song_index, n=playlist_length)
    my_songs_PCA = find_songs_by_genre_PCA(song_index, n=playlist_length)

    display_songs_2(my_songs)
    display_songs_2(my_songs_PCA)
