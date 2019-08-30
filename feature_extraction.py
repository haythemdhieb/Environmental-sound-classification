# coding= UTF-8

import glob
import os
import librosa
import numpy as np

def feature_extraction(file_name):
    y, sample_rate = librosa.load(file_name)
    

    # calcul des caractéristiques 
    
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=16000, n_mfcc=13,hop_length=256,n_fft=512), axis=1) 
    spec=np.mean( librosa.feature.melspectrogram(y=y, sr=16000, n_fft=512,hop_length=256,n_mels=13),axis=1)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    cent =np.mean( librosa.feature.spectral_centroid(y=y, sr=16000,hop_length=256,n_fft=512))
    flat=np.mean(librosa.feature.spectral_flatness(y=y, n_fft=512, hop_length=256))
   
    
    return mfccs,zcr,cent,flat,spec


# Process audio files: Return arrays with features and labels
def parse_audio_files(parent_dir, sub_dirs, file_ext='*.wav'):  ## le format de l'audio est wav
    features, labels = np.empty((0, 29)), np.empty(0)  # 29 features au  total.

    for label, sub_dir in enumerate(sub_dirs):  ##Enumerate(label varie de 1 à len(sub_dirs)  
        for file_name in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):  ##parent is audio-data, sub_dirs are audio classes
            try:
                mfccs,zcr,cent,flat,spec = feature_extraction(file_name)
            except Exception as e:
                print("[Error] there was an error in feature extraction. %s" % (e))
                continue

            extracted_features = np.hstack(
                [mfccs,zcr,cent,flat,spec]) 
            features = np.vstack([features, extracted_features])  
            labels = np.append(labels, label)
        print("Extracted features from %s, done" % (sub_dir))
    return np.array(features), np.array(labels, dtype=np.int)

# Read sub-directories (audio classes)
audio_directories = os.listdir('C:\\Users\\TOSHIBA\\Desktop\\basedonnée1\\audio')
audio_directories.sort()

# Function call to get labels and features
# This sabes a feat.npy and label.npy numpy-files in the current directory
features, labels = parse_audio_files('C:\\Users\\TOSHIBA\\Desktop\\basedonnée1\\audio', audio_directories)
np.save('features1.npy', features)
np.save('labels1.npy', labels)