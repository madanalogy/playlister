from genre_classifier.nnm import get_score, predict
from util.sentiment import extract
from playlist_generator.Nearest_Neighbours import find_songs_by_features, find_songs_by_valence

"""Ask user how long they want the playlist to be (n) and to pick the workflow

WF#1: Genre Classification + Sentiment Analysis
WF#2: Nearest Neighbours from SongId/Features
"""


def playlist_len():
    restrictions = "Please input an integer between 1-200"
    print("How long would you like your playlist to be? " + restrictions)
    length_input = input()
    try:
        length = int(length_input)
        if length < 1 or length > 200:
            print("Error: " + restrictions)
        else:
            return length
    except ValueError:
        print("Error: " + restrictions)
    return 0


def workflow_select():
    print("How would you like to generate your playlist? Please pick from the options below:")
    print("[1] Generate a playlist based on genre and mood")
    print("[2] Generate a playlist based on seed song(s)")
    workflow_input = input()
    if workflow_input != "1" or workflow_input != "2":
        print("Error: Please select an index from the list")
        return 0
    return int(workflow_input)


"""WF#1: Go through the entire process based on the steps below

WF#1-1: Ask user whether he/she wants to extract the genre from an audio file or pick the genre from a list
WF#1-1-AudioFile: Pass path to file to method predict
WF#1-1-PickGenre: Set genre variable based on input (preferably indexed)
WF#1-2: Ask user for text/phrase which should be passed to method extract
WF#1-3: Pass genre and sentiment to method find_songs_by_valence
"""


def workflow_1(playlist_length):
    # TODO: Implementation of WF#1
    return []


"""WF#2: Go through the entire process based on the steps below

WF#2-1: Ask user for the name of a song
WF#2-2: Search song list for matching songs, output a numbered list of songs
WF#2-3: User indicates index of item returned from search, song added to seed list
WF#2-4: Ask user if seed list complete, if no go back to #WF2-1 (threshold n-1)
WF#2-5: Pass seed list to method find_songs_by_features
"""


def workflow_2(playlist_length):
    # TODO: Implementation of WF#2
    return []


"""Go through entire sequence and print out generated playlist"""


def process():
    playlist_length = 0
    while playlist_length == 0:
        playlist_length = playlist_len()

    workflow = 0
    while workflow == 0:
        workflow = workflow_select()

    playlist = []
    if workflow == 1:
        playlist = workflow_1(playlist_length)
    elif workflow == 2:
        playlist = workflow_2(playlist_length)

    for song in playlist:
        print(song)


process()
