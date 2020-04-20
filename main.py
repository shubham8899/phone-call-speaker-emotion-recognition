import numpy as np
import pandas as pd
import tensorflow as tf
import keras.backend.tensorflow_backend
# import os
import matplotlib.pyplot as plt
import librosa
import wave
import keras
import librosa
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import rmsprop
from pyAudioAnalysis import audioSegmentation
from pydub import AudioSegment

from func import split, create_model_LSTM, split_audio_file, individual_emo, sp0_emo

predictor = {0:'neutral', 1:'calm', 2:'happy', 3:'sad', 4:'angry', 5:'fearful', 6:'disgust', 7:'surprised'}

np.set_printoptions(threshold=np.inf)
a = audioSegmentation.speakerDiarization('/home/shubham/Documents/SymbiosisHackathon/TataData/1.wav', 2, mt_step=0.05)

print(a)

b = split(a)
print(b)

split_audio_file(b, '/home/shubham/Documents/SymbiosisHackathon/TataData/1.wav')

# model = create_model_LSTM()
# model.load_weights('/home/shubham/Documents/SymbiosisHackathon/Model_A.h5')

# y, sr = librosa.load('/home/shubham/Documents/SymbiosisHackathon/newSong7.wav')
# mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T,axis=0)
# mfccs = mfccs.reshape((1,40,1))
# print(mfccs.shape)

j = sp0_emo(len(b))
print(j)



