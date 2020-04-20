import numpy as np
import pandas as pd
import tensorflow as tf
import keras.backend.tensorflow_backend
# import os
import matplotlib.pyplot as plt
import librosa
import wave
import keras
import pydub
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import rmsprop
from pyAudioAnalysis import audioSegmentation
from pydub import AudioSegment

predictor = {0:'neutral', 1:'calm', 2:'happy', 3:'sad', 4:'angry', 5:'fearful', 6:'disgust', 7:'surprised'}

def split(a):
    split = []
    cur=a[0]

    for i in range(len(a)):
        if(a[i]!=cur):
            split.append(i*100)
            cur=a[i]

    return split

def create_model_LSTM():

    model = Sequential()
    model.add(LSTM(128, return_sequences=False, input_shape=(40, 1)))
    model.add(Dense(64))
    model.add(Dropout(0.4))
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Dropout(0.4))
    model.add(Activation('relu'))
    model.add(Dense(8))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    return model

model = create_model_LSTM()
model.load_weights('Model_A.h5')    

def split_audio_file(split, filename):
    s = 0
    e = split[0]

    for i in range (1, len(split)):
        newAudio = AudioSegment.from_wav(filename)
        newAudio = newAudio[s:e]
        newAudio.export('diarized/diarized' + str(i) + '.wav', format="wav")
        s = e
        e = split[i]

def individual_emo(num):
    y, sr = librosa.load('diarized/diarized' + str(num) +'.wav')
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T,axis=0)
    mfccs = mfccs.reshape((1,40,1))
    a = model.predict(mfccs)
    a = a.reshape(8)
    a = np.array(a)
    print(predictor[list(a).index(np.max(a))])
    return list(a).index(np.max(a))

def sp0_emo(b):
    emo_sp0_arr = []
    for i in range(2, 15, 2):
        emo_sp0_arr.append(individual_emo(i))
    return emo_sp0_arr


# def combine_even(b):
#     sound_even = []
#     combined_sp0 = 0
    
#     for i in range (2, len(b)):
#         sound_even.append(AudioSegment.from_wav("newSong" + str(i) + ".wav"))
#         i = i+1

#     for i in range (2, len(b)):
#         combined_sp0 = combined_sp0 + sound_even[i]
#         i = i+1
    
#     combined_sp0.export("sp0.wav", format="wav")

# def combine_odd(b):
#     sound_odd = []
#     combined_sp1 = 0
    
#     for i in range (1, len(b)):
#         sound_odd.append(AudioSegment.from_wav("newSong" + str(i) + ".wav"))
#         i = i+1

#     for i in range (0, len(b)):
#         combined_sp1 = combined_sp1 + sound_odd[i]
#         i = i+1
    
#     combined_sp1.export("sp0.wav", format="wav")






# # m = create_model_LSTM()
# # m.load_weights('/home/shubham/Documents/SymbiosisHackathon/Model_A.h5')

