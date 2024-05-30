import librosa
import numpy as np
import pandas as pd
import os
# import compiam
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# save annotation columns and read annotations file 
annotation_columns = ['type', 'start', 'end', 'duration', 'svara']
annotations_path = os.path.join('svara_task','kamakshi_new.txt')
annotations = pd.read_csv(annotations_path, sep='\t', names=annotation_columns)

# Convert 'start' and 'end' columns to seconds
def time_to_seconds(time_str):
    h, m, s = map(float, time_str.split(':'))
    return h * 3600 + m * 60 + s

annotations['start'] = annotations['start'].apply(time_to_seconds)
annotations['end'] = annotations['end'].apply(time_to_seconds)


headers = ['id',
           'is_transition'
           ]

# create a dataframe
df_svara_transition = pd.DataFrame(columns = headers)

# accessing the audio file
audio_file = os.path.join('svara_task','separated_data','voice_separated.mp3')

# window size in seconds
window_size = 0.2

# load audio file
y, sr = librosa.load(audio_file, mono=True)

# calculate audio duration
audio_duration = librosa.get_duration(y=y, sr=sr)

# create windows of 0.5 seconds
windows = np.arange(0, audio_duration, window_size)

id = 0

for start_time in windows:
    end_time = start_time + window_size

    # check if there is a transition between svaras in the current window
    is_transition = 0
    for index, row in annotations.iterrows():
        if (row['start'] >= start_time and row['start'] < end_time) or (row['end'] >= start_time and row['end'] < end_time):
            is_transition = 1
            break

    # add the information to the DataFrame
    df_svara_transition = df_svara_transition.append({'id': id, 'is_transition': is_transition}, ignore_index=True)
    id += 1


df_svara_transition.to_csv('svara_transition/svara_transitions.csv', index=False)
