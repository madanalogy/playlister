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

data_file_name = os.path.dirname(os.path.abspath(__file__)) + "/songs_with_genres.csv"
data = pd.read_csv(data_file_name)

# spotify_song_names_file_name = os.path.dirname(os.path.abspath(__file__)) + "/spotify_song_names.csv"
# spotify_song_names = pd.read_csv(spotify_song_names_file_name)['simple_name']
spotify_song_names = data['simple_name'].sort_values()

# print(spotify_song_names.iloc[0])

# spotify_song_names = data['simple_name'].sort_values()

# spotify_songs_by_name = {}
# for i, song in data.iterrows():
#     spotify_songs_by_name[song.simple_name] = (i, song)

song_index_by_name = {simple_name: i for i, simple_name in data['simple_name'].iteritems()}

# song_index_by_name = {}
# for i, song in data.iterrows():
#     song_index_by_name[song.simple_name] = i


# def filter_data(data, column, value):
#     return data.loc[[(value in row) for row in data[column]]]


def find_spotify_info(song_name):
    return data.iloc[song_index_by_name[song_name]]


def find_song_index(song_name):
    return song_index_by_name[song_name]
    # if song_name not in spotify_songs_by_name:
    #     return None
    # return spotify_songs_by_name[song_name]


def find_songs_by_keyword(keyword):
    songs = []
    for song_name in spotify_song_names:
        if keyword in song_name:
            songs.append(song_name)
    return songs


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


def find_songs_by_features(seeds, n=10, pca=True, components=7):
    x = data[features]
    x_scaled = StandardScaler().fit_transform(x)

    # TODO: Catch ID not in song list, retrieve features from Spotify
    # song_index = data.loc[data['id'] == song_id].index[0]

    if pca:
        pca = PCA(n_components=components)
        principal_components = pca.fit_transform(x_scaled)
        rows_with_seeds = principal_components[seeds]
        neighbors = NearestNeighbors(n_neighbors=n + 1, algorithm='ball_tree').fit(principal_components)
    else:
        rows_with_seeds = x_scaled[seeds]
        neighbors = NearestNeighbors(n_neighbors=n + 1, algorithm='ball_tree').fit(x_scaled)

    distances, indices = neighbors.kneighbors(np.array(rows_with_seeds))

    songs = []
    for song_index in indices[0]:
        songs.append(data.iloc[song_index])

    return songs


def find_songs_by_valence(genre, valence, n=10):
    data = pd.read_csv('popular_' + genre + '_songs.csv')
    diff = []
    for i, song in data.iterrows():
        diff.append(abs(song['valence'] - valence))

    data['diff'] = diff

    data = data.sort_values(['diff', 'popularity'], ascending=[True, False])
    songs = data.head(n)
    return songs


# if __name__ == '__main__':
#     workflow = int(input("Choose a workflow: "))
#     playlist_length = int(input("Enter the length of the playlist (must be less than 3000): "))

#     if workflow == 1:
#         # Workflow 1
#         genre = input("Input a genre: ")
#         valence = float(input("Input a valence: "))

#         songs = find_songs_by_valence(genre, valence, n=playlist_length)
#         display_songs_1(songs, genre, valence)

#     elif workflow == 2:
#         # Workflow 2
#         song_id = input("Input a Spotify ID: ")

#         my_songs = find_songs_by_features(song_id, n=playlist_length, pca=False)
#         my_songs_PCA = find_songs_by_features(song_id, n=playlist_length, pca=True)

#         display_songs_2(my_songs)
#         display_songs_2(my_songs_PCA)
