from genre_classifier.nnm import get_score, predict

# Some simply scripts to run predict
print(get_score())
print(predict(['./data_audio/genres_original/blues/blues.00000.wav', './data_audio/genres_original/blues/blues.00001.wav']))