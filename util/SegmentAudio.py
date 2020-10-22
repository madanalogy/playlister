from pydub import AudioSegment
import os, sys
song_folder_directory = None
song_folder_output_directory = None
def usage():
    print("Please specify the directory of the songs dataset and where to output the dataset. E.g. ./test/original/blues ../test/30sec/blues")

try:
    song_folder_directory = os.path.abspath(sys.argv[1])
    song_folder_output_directory = os.path.abspath(sys.argv[2])
except IndexError:
    usage()
    sys.exit(2)

songs = os.listdir(song_folder_directory)
split_interval = 5
sampling_indices = [1,2,3]
sample_duration = 30 # 30 seconds sampling duration
for song in songs:
    song_file_path = os.path.join(song_folder_directory, song)
    song_name, song_ext = os.path.splitext(song)
    audio = AudioSegment.from_wav(song_file_path)
    song_duration = audio.duration_seconds

    interval_length = int(song_duration / split_interval)
    for i in sampling_indices:
        offset_start = interval_length * i
        offset_end = offset_start + 30
        millisecond_multiplier = 1000
        new_audio = audio[offset_start * millisecond_multiplier: offset_end * millisecond_multiplier]
        new_audio.export(os.path.join(song_folder_output_directory, song_name + "{}".format(i) +  song_ext), format='wav')
