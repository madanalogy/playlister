from genre_classifier.nnm import get_score, predict
from util.sentiment import extract
from playlist_generator.Nearest_Neighbours import find_songs_by_features, find_songs_by_valence

"""Ask user how long they want the playlist to be (n) and to pick the workflow

WF#1: Genre Classification + Sentiment Analysis
WF#2: Nearest Neighbours from SongId/Features
"""

"""WF#1: Go through the entire process based on the steps below

WF#1-1: Ask user whether he/she wants to extract the genre from an audio file or pick the genre from a list
WF#1-1-AudioFile: Pass path to file to method predict
WF#1-1-PickGenre: Set genre variable based on input (preferably indexed)
WF#1-2: Ask user for text/phrase which should be passed to method extract
WF#1-3: Pass genre and sentiment to method find_songs_by_valence
"""

"""WF#2: Go through the entire process based on the steps below

WF#2-1: Ask user for the name of a song
WF#2-2: Search song list for matching songs, output a numbered list of songs
WF#2-3: User indicates index of item returned from search, song added to seed list
WF#2-4: Ask user if seed list complete, if no go back to #WF2-1 (threshold n-1)
WF#2-5: Pass seed list to method find_songs_by_features
"""

"""Print out list of songs generated for playlist"""

