from genre_classifier.nnm import predict
from util.sentiment import extract
# from playlist_generator.Nearest_Neighbours import find_songs_by_features, find_songs_by_valence, major_genres
from bisect import insort
"""
Initialization:
workflow_1_data =
"""

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
    if workflow_input != "1" and workflow_input != "2":
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


def select_genre_method():
    print("How should the genre of your playlist be decided? Please pick from the options below:")
    print("[1] Extract genre from an audio file")
    print("[2] Select genre from predefined list")
    genre_select = input()
    if genre_select != "1" or genre_select != "2":
        print("Error: Please select an index from the list")
        return 0
    return int(genre_select)


def select_genre_list():
    err = "Error: Please input an integer from the list"
    print("Please pick a genre from the list below:")
    for idx, genre in enumerate(major_genres):
        print("[" + str(idx) + "] " + genre)
    genre_select = input()
    try:
        genre_index = int(genre_select)
        if 0 <= genre_index < len(major_genres):
            return genre_index
    except ValueError:
        pass
    print(err)
    return -1


def workflow_1(playlist_length):
    genre_select = 0
    while genre_select == 0:
        genre_select = select_genre_method()

    genre = ""
    if genre_select == 1:
        genre = predict(input("Please input path to audio file: "))
    elif genre_select == 2:
        genre_id = -1
        while genre_id == -1:
            genre_id = select_genre_list()
        genre = major_genres[genre_id]

    valence = extract(input("How would you like your playlist to feel?: "))

    return find_songs_by_valence(genre, valence, playlist_length)


"""WF#2: Go through the entire process based on the steps below

WF#2-1: Ask user for the name of a song
WF#2-2: Search song list for matching songs, output a numbered list of songs
WF#2-3: User indicates index of item returned from search, song added to seed list
WF#2-4: Ask user if seed list complete, if no go back to #WF2-1 (threshold n-1)
WF#2-5: Pass seed list to method find_songs_by_features
"""

songs = [
    "Patato",
    "Petato",
    "Pitato",
    "Potato",
    "Putato",
    "Poteto",
    "Potito",
    "Pototo",
    "Potuto",
    "Potata",
    "Potate",
    "Potati",
    "Potatu",
]

songs = list('abcdefghijklmnopqrstuvwxyz')

songs = [song.strip().lower() for song in songs]

def select_option(options, wrap=0):
    num_options = len(options)

    for i, option in enumerate(options, start=1):
        print(f'[{i}] {option}')

    option = int(input()) # TODO: error checking
    valid_option = option > 0 and option <= num_options

    return (option, valid_option)

def input_bool(message):
    print(message)
    while True:
        print('[y] Yes')
        print('[n] No')
        try_again = input().strip().lower() # TODO: error checking
        if try_again == 'y':
            return True
        if try_again == 'n':
            return False
        print('What was that?')

def find_seed_songs(songs=songs):
    matching_songs = []

    finding_songs = True
    while finding_songs:
        print('Find songs containing the keyword: ', end='')
        keyword = input().strip().lower()

        for song in songs:
            if keyword in song:
                matching_songs.append(song)

        if matching_songs:
            finding_songs = False
        else:
            print(f'Sorry. No songs matched the keyword "{keyword}".')
            finding_songs = input_bool('Try another keyword?')

    return matching_songs

def select_seed_song(songs):
    song = None

    selecting_song = True
    while selecting_song:
        print('Select the index of the song that you wish to add: ')
        (index, valid_index) = select_option(songs)

        if not valid_index:
            print(f'Sorry. There is no song at index "{index}".')
            selecting_song = input_bool("Try another index?")

        else:
            song = songs[index - 1]
            selecting_song = False

    return song

def add_seed(seeds):
    unique_seeds = set(seeds)

    running = True
    while running:
        # Find all songs matching a keyword
        songs = find_seed_songs()
        if not songs:
            break

        # Select a song from the search results
        song = select_seed_song(songs)
        if song is not None:
            unique_seeds.add(song)
            print(f'Added: {song}')

        running = input_bool('Add another song?:')

    return sorted(unique_seeds)

def remove_seed(seeds):
    updated_seeds = seeds
    running = True

    while running:
        print('Select the index of the song that you wish to remove: ')
        (index, valid_index) = select_option(seeds)

        if not valid_index:
            print(f'Sorry. There is no song at index "{index}".')
            running = input_bool("Try another index?")

        else:
            song = updated_seeds.pop(index - 1)
            print(f'Removed: {song}')

            if not updated_seeds:
                print('No more songs to remove.')
                running = False
            else:
                running = input_bool('Remove another song?')

    return updated_seeds

def workflow_2(playlist_length):
    seeds = []
    playlist_songs = []
    finished_workflow = False

    while not finished_workflow:
        print(f'Seeds: {seeds}')
        print('Please pick from the options below:')
        print('[1] Add song')
        if seeds:
            print('[2] Remove song')
            print('[3] Generate playlist')
        print('[0] Cancel')

        option = input()

        if option == '0':
            return None
        if option == '1':
            seeds = add_seed(seeds)
        elif seeds:
            if option == '2':
                seeds = remove_seed(seeds)
            elif option == '3':
                #playlist_songs = generate_playlist(list(seeds))
                finished_workflow = True
        else:
            print('Invalid option. Please try again.')

    return playlist_songs


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

    if playlist is None:
        print('Cancelled')
    else:
        for song in playlist:
            print(song)

    print('FINISHED')


process()
