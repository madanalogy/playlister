import os
from pydub import AudioSegment

class FileConversion(object):
    def __init__(self, filename):
        self.filename = filename

    def convert(self):
        fn, ext = os.path.splitext(self.filename)
        if ext != ".wav" and ext != ".mp3":
            print("file format not supported. please provide .mp3 or .wav audio files")
            return None

        if ext == ".mp3":
            # Convert to .wav file
            print("File format is .mp3. Converting to .wav")
            sound = AudioSegment.from_mp3(self.filename)
            sound.export(fn + ".wav", format="wav")
        return fn + ".wav"