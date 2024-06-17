import os
import numpy as np
import pandas as pd
import librosa
from sklearn.preprocessing import StandardScaler

# Headers for the DataFrame
headers = ['id',
           'max pitch', 
           'min pitch', 
           'mean pitch', 
           'num change points', 
           'standard deviation',
           'zero cross',
           'loudness',
           'amplitude envelope',
           'band energy ratio',
           'spectral centroid',
           'spectral bandwidth',
           'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5',
           'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10',
           'mfcc11', 'mfcc12', 'mfcc13',
           'prev_sa', 'prev_ri', 'prev_ga', 'prev_ma', 'prev_pa', 'prev_dha', 'prev_ni',
           'next_sa', 'next_ri', 'next_ga', 'next_ma', 'next_pa', 'next_dha', 'next_ni',
           'is_sa',
           'is_ri',
           'is_pa',
           'is_ni',
           'is_ma',
           'is_ga',
           'is_dha']

# Create a DataFrame to store features
df_features = pd.DataFrame(columns=headers)

# Load the entire audio file
audio_file = os.path.join('svara_task', 'separated_data', 'voice_separated.mp3')
y, sr = librosa.load(audio_file, mono=True)

# Read the annotations file
annotation_columns = ['type', 'start', 'end', 'duration', 'svara']
annotations_path = os.path.join('svara_task', 'kamakshi_new.txt')
annotations = pd.read_csv(annotations_path, delim_whitespace=True, header=None, names=annotation_columns)

# Convert 'start' and 'end' columns to seconds
def time_to_seconds(time_str):
    h, m, s = map(float, time_str.split(':'))
    return h * 3600 + m * 60 + s

annotations['start'] = annotations['start'].apply(time_to_seconds)
annotations['end'] = annotations['end'].apply(time_to_seconds)

# Initialize id
id = 1

# Iterate through each row in the annotations file
for _, row in annotations.iterrows():
    start = row['start']
    end = row['end']
    svara_type = row['svara']
    
    # Extract the segment of the audio corresponding to the svara
    y_svara = y[int(start * sr):int(end * sr)]
    
    # Extract the pitch
    pitches, magnitudes = librosa.piptrack(y=y_svara, sr=sr)
    
    # Filter the valid frequencies
    valid_pitches = pitches[pitches > 0]
    
    # PITCH CURVE FEATURES
    max_pitch = np.max(valid_pitches) if len(valid_pitches) > 0 else 0
    min_pitch = np.min(valid_pitches) if len(valid_pitches) > 0 else 0
    mean_pitch = np.mean(valid_pitches) if len(valid_pitches) > 0 else 0
    std = np.std(valid_pitches) if len(valid_pitches) > 0 else 0
    
    # Number of Change Points
    second_derivative = np.diff(np.diff(magnitudes))
    num_change_points = np.sum(second_derivative[:-1] * second_derivative[1:] < 0)
    
    # TIME DOMAIN FEATURES
    splits = np.array_split(y_svara, 200)
    max_values = [np.max(np.abs(split)) for split in splits]
    mean_amplitude_envelope = np.mean(max_values)
    loudness = np.mean(librosa.feature.rms(y=y_svara)[0])
    zero_cross = np.mean(librosa.feature.zero_crossing_rate(y=y_svara)[0])
    
    # FREQUENCY DOMAIN FEATURES
    stft = np.abs(librosa.stft(y_svara))
    ber = np.sum(stft[:stft.shape[0] // 2, :], axis=0) / np.sum(stft[stft.shape[0] // 2:, :], axis=0)
    band_energy_ratio = np.mean(ber)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y_svara, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y_svara, sr=sr))
    
    # Mel-Frequency Cepstral Coefficients (MFCCs)
    mfccs = librosa.feature.mfcc(y=y_svara, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    
    # Append the features to the dataframe
    df_features = df_features.append({'id': id,
                                      'max pitch': max_pitch, 
                                      'min pitch': min_pitch, 
                                      'mean pitch': mean_pitch, 
                                      'num change points': num_change_points,
                                      'standard deviation': std,
                                      'zero cross': zero_cross,
                                      'loudness': loudness,
                                      'amplitude envelope': mean_amplitude_envelope,
                                      'band energy ratio': band_energy_ratio,
                                      'spectral centroid': spectral_centroid,
                                      'spectral bandwidth': spectral_bandwidth,
                                      'mfcc1': mfccs_mean[0], 'mfcc2': mfccs_mean[1], 'mfcc3': mfccs_mean[2],
                                      'mfcc4': mfccs_mean[3], 'mfcc5': mfccs_mean[4], 'mfcc6': mfccs_mean[5],
                                      'mfcc7': mfccs_mean[6], 'mfcc8': mfccs_mean[7], 'mfcc9': mfccs_mean[8],
                                      'mfcc10': mfccs_mean[9], 'mfcc11': mfccs_mean[10], 'mfcc12': mfccs_mean[11],
                                      'mfcc13': mfccs_mean[12],
                                      'prev_sa': 0, 'prev_ri': 0, 'prev_ga': 0, 'prev_ma': 0, 'prev_pa': 0, 'prev_dha': 0, 'prev_ni': 0,
                                      'next_sa': 0, 'next_ri': 0, 'next_ga': 0, 'next_ma': 0, 'next_pa': 0, 'next_dha': 0, 'next_ni': 0,
                                      'is_sa': 1 if svara_type == 'sa' else 0,
                                      'is_ri': 1 if svara_type == 'ri' else 0,
                                      'is_pa': 1 if svara_type == 'pa' else 0,
                                      'is_ni': 1 if svara_type == 'ni' else 0,
                                      'is_ma': 1 if svara_type == 'ma' else 0,
                                      'is_ga': 1 if svara_type == 'ga' else 0,
                                      'is_dha': 1 if svara_type == 'dha' else 0,
                                      }, ignore_index=True)
    
    # Increment the id
    id += 1

# Add previous and next svara features
for i in range(len(df_features)):
    if i > 0:
        prev_svara_type = df_features.iloc[i - 1][['is_sa', 'is_ri', 'is_ga', 'is_ma', 'is_pa', 'is_dha', 'is_ni']]
        for svara in prev_svara_type.index:
            if prev_svara_type[svara] == 1:
                df_features.loc[df_features.index[i], f'prev_{svara.split("_")[1]}'] = 1
    else:
        for svara in ['sa', 'ri', 'ga', 'ma', 'pa', 'dha', 'ni']:
            df_features.loc[df_features.index[i], f'prev_{svara}'] = 0

    if i < len(df_features) - 1:
        next_svara_type = df_features.iloc[i + 1][['is_sa', 'is_ri', 'is_ga', 'is_ma', 'is_pa', 'is_dha', 'is_ni']]
        for svara in next_svara_type.index:
            if next_svara_type[svara] == 1:
                df_features.loc[df_features.index[i], f'next_{svara.split("_")[1]}'] = 1
    else:
        for svara in ['sa', 'ri', 'ga', 'ma', 'pa', 'dha', 'ni']:
            df_features.loc[df_features.index[i], f'next_{svara}'] = 0

# Normalization using Z-score
scaler = StandardScaler()

# Separate the 'id' and svara type columns from the features to be normalized
ids = df_features['id']
svara_types = df_features[['is_sa', 'is_ri', 'is_pa', 'is_ni', 'is_ma', 'is_ga', 'is_dha']]
features_to_normalize = df_features.drop(columns=['id', 'is_sa', 'is_ri', 'is_pa', 'is_ni', 'is_ma', 'is_ga', 'is_dha'])

# Apply Z-score normalization
normalized_features = scaler.fit_transform(features_to_normalize)

# Recreate the DataFrame with normalized features
df_normalized = pd.DataFrame(normalized_features, columns=features_to_normalize.columns)

# Add back the 'id' and svara type columns
df_normalized['id'] = ids
df_normalized[['is_sa', 'is_ri', 'is_pa', 'is_ni', 'is_ma', 'is_ga', 'is_dha']] = svara_types

# Ensure the 'id' column is the first column
df_normalized = df_normalized[['id'] + features_to_normalize.columns.tolist() + ['is_sa', 'is_ri', 'is_pa', 'is_ni', 'is_ma', 'is_ga', 'is_dha']]

# Save the normalized dataframe to a CSV file
df_normalized.sort_values(by='id', inplace=True)
df_normalized.to_csv('svara_transition/svara_transitions_method_2.csv', index=False)


