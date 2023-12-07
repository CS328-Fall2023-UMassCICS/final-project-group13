# Final Project Python File
import numpy as np
import pandas as pd
import glob
import sys
import os
import librosa 
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
#import librosa.display

from sklearn import tree, metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# using functions from the Assignment 4 code 

# extract features 
def extract_features(data):
    zero_crossings = librosa.feature.zero_crossing_rate(y=data)
    mfcc = librosa.feature.mfcc(y=data, sr=sample_rate)
    rootms = librosa.feature.rms(y=data)
    mel = librosa.feature.melspectrogram(y=data, sr=sample_rate)
    # might need more parameters?
    spectral_contrast = librosa.feature.spectral_contrast(y=data)
    # Average across columns (axis=1)
    zcr_avg = np.mean(zero_crossings, axis=1)
    # not sure if I need the mfcc part 
    mfcc_avg = np.mean(mfcc, axis=1)
    rms_avg = np.mean(rootms, axis=1) 
    mel_avg = np.mean(mel, axis=1)
    spc_avg = np.mean(spectral_contrast, axis=1)
 
    # Concatenate into single row 
    features = np.concatenate([zcr_avg, mfcc_avg, rms_avg, mel_avg, spc_avg])
    # Convert to dataframe and transpose it so that the shape is 1x150
    df = pd.DataFrame(features).T
    return df

#train decision tree 
# error because of lack of data here?
def train_random_forest(frames):
    # Use pandas iloc fn to extract the first 150 columns as features.
    # Careful about how the indexing works (cols start from 0)
    X = frames.iloc[: , 0:156]
    # Use pandas iloc function to extract the 151st column as the prediction target.
    # Again, careful about how indexing works (col numbers start from 0)
    # this needs to be the labels?
    y = frames.iloc[: , 157]
    # Split data
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    Emotion_rf = RandomForestClassifier()
    Emotion_rf_model = Emotion_rf.fit(X_train, y_train)
    Emotion_rf_pred = Emotion_rf_model.predict(X_test)
    print(classification_report(y_test, Emotion_rf_pred))
    # Evaluate on test set
    acc = Emotion_rf_model.score(X_test, y_test)
    Emotion_rf_cm = confusion_matrix(y_test, Emotion_rf_pred)
    print(Emotion_rf_cm)
    return Emotion_rf_model, Emotion_rf_cm, acc



# # possibly combine all data to csv?
# def all_data_to_combined_csv():
#     path = "AudioFiles/**/*.csv"
#     list_files=glob.glob(path, recursive=True)
#     all_data = pd.DataFrame()
#     for file in list_files:
#         # check this 
#         activity = file.split('\\')[1]
#         activity = activity_file = os.path.basename(file)
#         data = pd.read_csv(file)
#         data = calc_magnitude(data)
#         data = remove_noise(data, 100)
#         data = extract_features(data, 10, 100, activity)
#         all_data = pd.concat([all_data, data])

#     all_data.to_csv("all_data.csv")


# code to call it 
filenames = glob.glob("AudioFiles/*/*.wav")
frames = pd.DataFrame()
# change offset and duration for loading the file to split into 10 second intervals 
for filename in filenames:
   sound = filename.split('\\')[1]
   data, sample_rate = librosa.load(filename)
   # show the graph
   #fig, ax = plt.subplots(nrows =3, sharex = True)
   # figure out how to get plots to be the same scale 
   librosa.display.waveshow(data, sr=sample_rate, color="blue")
   feature_df = extract_features(data)
   sound_df = pd.DataFrame([sound])
   combined_df = pd.concat([feature_df, sound_df], axis = 1)
   frames = pd.concat([combined_df, frames])

# 158 because i added another feature 
col_names = [f'feat_{i}' for i in range (157)] + ['label']
frames.columns = col_names
Sound_rf_model, sound_rf_cm, acc = train_random_forest(frames)
X = 5