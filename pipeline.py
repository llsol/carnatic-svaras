import librosa
import numpy as np
import pandas as pd
import os
import seaborn
import compiam


# Data Curation

# source separation

# svara segmentation



# Feature extraction

# create a header array for the dataframe
headers = ['seg num',
           'max pitch', 
           'min pitch', 
           'pitch range', 
           'mean pitch', 
           'avg start pitch', 
           'avg end pitch', 
           'num change points', 
           'prec svara', 
           'succ svara', 
           'prec+succ svara', 
           'prec dir', 
           'succ dir', 
           'prec+succ dir']

# create a dataframe
df_features = pd.DataFrame(columns = headers)

# accessing the audio files
single_svara_path = os.path.join('single_svara')

# iterate through the audio files to extract the current state-of-the-art selected pitch features
for file in os.listdir(single_svara_path):
    # load the audio file
    y, sr = librosa.load(os.path.join(single_svara_path, file))
    
    # extract the pitch
    pitches, magnitudes = librosa.piptrack(y = y, sr = sr)
    
    # extract the pitch features
    max_pitch = np.max(pitches)
    min_pitch = np.min(pitches)
    pitch_range = max_pitch - min_pitch
    mean_pitch = np.mean(pitches)
    avg_start_pitch = np.mean(pitches[0])
    avg_end_pitch = np.mean(pitches[-1])
    num_change_points = None


    # split the audio file name to extract the segment number
    seg_num = file.split('_')[1]
    seg_num = seg_num.split('.')[0]


    
    # append the features to the dataframe
    df_features = df_features.append({'seg num': seg_num,
                                      'max pitch': max_pitch, 
                                      'min pitch': min_pitch, 
                                      'pitch_range': pitch_range, 
                                      'mean pitch': mean_pitch, 
                                      'avg start pitch': avg_start_pitch, 
                                      'avg end pitch': avg_end_pitch, 
                                      'num change points': num_change_points, 
                                      }, ignore_index = True)

# state-of-the-art numerical melodic features


# state-of-the-art context features

# preceding svara
# succeeding svara
# precedent-succeeding svara
# preceding direction
# succeeding direction
# preceding-succeeding direction



# Modelling
#sklearn



# Evaluation
#accuracy 



# GUI
#(qt design), (streamlit)
