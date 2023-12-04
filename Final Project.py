# Final Project Python File
import numpy as np
import pandas as pd
import glob
import sys
import os
import librosa 

# calculate magnitutde functon
# this might be different now 
def calc_magnitude(data):
    data['audio_mag'] = np.sqrt(data['x']**2 + data['y']**2 + data['z']**2)
    data['audio_mag'] = data['audio_mag'] - data['audio_mag'].mean()

    return data 

# remove noise 

# add features 

# extract features 

#train decision tree 

#create a classifier
    # not positive that we do this 


# possibly combine all data to csv?
def all_data_to_combined_csv():
    path = "AudioFiles/**/*.csv"
    list_files=glob.glob(path, recursive=True)
    all_data = pd.DataFrame()
    for file in list_files:
        # check this 
        activity = file.split('\\')[1]
        activity = activity_file = os.path.basename(file)
        data = pd.read_csv(file)
        data = calc_magnitude(data)
        data = remove_noise(data, 100)
        data = extract_features(data, 10, 100, activity)
        all_data = pd.concat([all_data, data])

    all_data.to_csv("all_data.csv")

