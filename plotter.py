# from genre_classifier.nnm import plot_cnf_mat, get_score
import librosa
import librosa.display as lbdp
import numpy as np
import matplotlib.pyplot as plt

plot_cnf_mat()
# print(get_score())
# Used metal0073, metal0078
# And 
sample_audio_file1 = './data_audio/genres_original/metal/metal.00073.wav'
sample_audio_file2 = './data_audio/genres_original/pop/pop.00052.wav'

s1_y, s1_sr = librosa.load(sample_audio_file1)
s2_y, s2_sr = librosa.load(sample_audio_file2)

s1_mfccs = librosa.feature.mfcc(s1_y, sr=s1_sr)
s2_mfccs = librosa.feature.mfcc(s2_y, sr=s2_sr)

s1_chroma_stft = librosa.feature.chroma_stft(s1_y, sr=s1_sr)
s2_stft = librosa.feature.chroma_stft(s2_y, sr=s2_sr)

s1_spect = librosa.feature.melspectrogram(y=s1_y, sr=s1_sr)
s2_spect = librosa.feature.melspectrogram(y=s2_y, sr=s2_sr)

fig = plt.figure()
plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=None, hspace=None)
ax3 = fig.add_subplot(221)
lbdp.specshow(librosa.amplitude_to_db(s1_spect, ref=np.max), sr=s1_sr, x_axis='time')
plt.colorbar(format='%+2.0f dB')
ax3.title.set_text('metal.0073.wav - melspectrogram')
ax4 = fig.add_subplot(222)
ax4.title.set_text('pop.0052.wav - melspectrogram')
lbdp.specshow(librosa.amplitude_to_db(s2_spect, ref=np.max), sr=s2_sr, x_axis='time')
plt.colorbar(format='%+2.0f dB')

ax1 = fig.add_subplot(223)
ax1.title.set_text('metal.0073.wav - mfcc')
lbdp.specshow(s1_mfccs, sr=s1_sr, x_axis='time')
plt.colorbar()
ax2 = fig.add_subplot(224)
ax2.title.set_text('pop.00052.wav - mfcc')
lbdp.specshow(s2_mfccs, sr=s2_sr, x_axis='time')
plt.colorbar()

plt.show()

