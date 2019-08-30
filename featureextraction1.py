# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 23:04:06 2019

@author: Haythem
"""
import os
import librosa
import matplotlib.pyplot as plt
import  librosa.display 
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')


audio_path = 'C:\\Users\\TOSHIBA\\Desktop\\basedonnée1\\audio\\10-fire cracking'
pict_Path = 'C:\\Users\\TOSHIBA\\Desktop\\basedonnée1\\data\\firecracking'

def wav2img(wav_path, targetdir='', figsize=(10,6)):
    y,sr = librosa.load(wav_path)
    fig=plt.figure(figsize=(8, 6))
    mfccs = librosa.feature.mfcc(y=y, sr=16000, n_mfcc=13,hop_length=256,n_fft=512)
    mfccs=librosa.display.specshow(mfccs)

    output_file = wav_path.split('/')[-1].split('.wav')[0]
    output_file = targetdir +'/'+ output_file
    fig.savefig(output_file+'.png',bbox_inches='tight' )
    plt.close()

#for i, x in enumerate(subFolderList):
all_files = [y for y in os.listdir(audio_path) if '.wav' in y]
for file in all_files:
        wav2img(audio_path + '/' + file, pict_Path)