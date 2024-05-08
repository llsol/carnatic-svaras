import librosa
import numpy as np
import pandas as pd
import os
import compiam
import matplotlib.pyplot as plt
import seaborn as sns


# Data Curation

# source separation

# svara segmentation



# Feature extraction

# create a header array for the dataframe
headers = ['id',
           'max pitch', 
           'min pitch', 
           'mean pitch', 
           'num change points', 
           'standard deviation',
           'zero cross',
           'loudness']

svaras = ['sa', 'ri', 'pa', 'ni', 'ma', 'ga', 'dha']

for svara in svaras:
    

    # create a dataframe
    df_features = pd.DataFrame(columns = headers)

    # accessing the audio files
    single_svara_path = f'single_svara/{svara}'
    id = 0


    # iterate through the audio files to extract the current state-of-the-art selected pitch features
    for file in os.listdir(single_svara_path):
        # load the audio file
        y, sr = librosa.load(os.path.join(single_svara_path, file))
        # extract the pitch
        pitches, magnitudes = librosa.piptrack(y = y, sr = sr)

        # filtrem les frequencies valides:
        valid_pitches = pitches[pitches>0]
        

        # extract the pitch features
        max_pitch = np.max(valid_pitches)
        min_pitch = np.min(valid_pitches)

        pitch_range = max_pitch - min_pitch
        mean_pitch = np.mean(valid_pitches)
        avg_start_pitch = np.mean(valid_pitches[0])
        avg_end_pitch = np.mean(valid_pitches[-1])

        second_derivative = np.diff(np.diff(magnitudes))
        num_change_points =  np.sum(second_derivative[:-1] * second_derivative[1:] < 0)
        
        loudness = np.sqrt(np.mean(magnitudes**2))

        std = np.std(valid_pitches)

        zero_cross = np.sum(np.diff(magnitudes) * magnitudes[:,:-1] < 0)
        

        # cuenta el numero de svaras que hay:
        id += 1

        
        # append the features to the dataframe
        df_features = df_features.append({'id': id,
                                        'max pitch': max_pitch, 
                                        'min pitch': min_pitch, 
                                        'mean pitch': mean_pitch, 
                                        'num change points': num_change_points,
                                        'standard deviation': std,
                                        'zero cross': zero_cross,
                                        'loudness': loudness 
                                        }, ignore_index = True)


    df_features.to_csv(f'features/{svara}_features.csv',index=False)
    
    
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
