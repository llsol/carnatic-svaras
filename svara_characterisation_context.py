import os
import numpy as np
import pandas as pd
import librosa
from sklearn.preprocessing import StandardScaler

# Headers for the DataFrame
headers = ['id',
           'max pitch', 'max pitch prev', 'max pitch post',
           'min pitch', 'min pitch prev', 'min pitch post',
           'mean pitch', 'mean pitch prev', 'mean pitch post',
           'num change points', 'num change points prev', 'num change points post',
           'standard deviation', 'standard deviation prev', 'standard deviation post',
           'zero cross', 'zero cross prev', 'zero cross post',
           'loudness', 'loudness prev', 'loudness post',
           'amplitude envelope', 'amplitude envelope prev', 'amplitude envelope post',
           'band energy ratio', 'band energy ratio prev', 'band energy ratio post',
           'spectral centroid', 'spectral centroid prev', 'spectral centroid post',
           'spectral bandwidth', 'spectral bandwidth prev', 'spectral bandwidth post',
           'mfcc1', 'mfcc1 prev', 'mfcc1 post', 'mfcc2', 'mfcc2 prev', 'mfcc2 post',
           'mfcc3', 'mfcc3 prev', 'mfcc3 post', 'mfcc4', 'mfcc4 prev', 'mfcc4 post',
           'mfcc5', 'mfcc5 prev', 'mfcc5 post', 'mfcc6', 'mfcc6 prev', 'mfcc6 post',
           'mfcc7', 'mfcc7 prev', 'mfcc7 post', 'mfcc8', 'mfcc8 prev', 'mfcc8 post',
           'mfcc9', 'mfcc9 prev', 'mfcc9 post', 'mfcc10', 'mfcc10 prev', 'mfcc10 post',
           'mfcc11', 'mfcc11 prev', 'mfcc11 post', 'mfcc12', 'mfcc12 prev', 'mfcc12 post',
           'mfcc13', 'mfcc13 prev', 'mfcc13 post',
           'is_sa', 'is_ri', 'is_pa', 'is_ni', 'is_ma', 'is_ga', 'is_dha']

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

# Helper function to extract features from a given audio segment
def extract_features(y_segment, sr):
    pitches, magnitudes = librosa.piptrack(y=y_segment, sr=sr)
    valid_pitches = pitches[pitches > 0]
    if len(valid_pitches) == 0:
        return [0] * 13 + [0] * 13  # Return zeros if no valid pitches are found
    
    max_pitch = np.max(valid_pitches)
    min_pitch = np.min(valid_pitches)
    mean_pitch = np.mean(valid_pitches)
    std = np.std(valid_pitches)
    second_derivative = np.diff(np.diff(magnitudes))
    num_change_points = np.sum(second_derivative[:-1] * second_derivative[1:] < 0)
    splits = np.array_split(y_segment, 200)
    max_values = [np.max(np.abs(split)) for split in splits]
    mean_amplitude_envelope = np.mean(max_values)
    loudness = np.mean(librosa.feature.rms(y=y_segment)[0])
    zero_cross = np.mean(librosa.feature.zero_crossing_rate(y=y_segment)[0])
    stft = np.abs(librosa.stft(y_segment))
    ber = np.sum(stft[:stft.shape[0] // 2, :], axis=0) / np.sum(stft[stft.shape[0] // 2:, :], axis=0)
    band_energy_ratio = np.mean(ber)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y_segment, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y_segment, sr=sr))
    mfccs = librosa.feature.mfcc(y=y_segment, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    
    return [max_pitch, min_pitch, mean_pitch, num_change_points, std, zero_cross, loudness, mean_amplitude_envelope, band_energy_ratio, spectral_centroid, spectral_bandwidth] + list(mfccs_mean)

# Iterate through each row in the annotations file
for _, row in annotations.iterrows():
    start = row['start']
    end = row['end']
    svara_type = row['svara']
    duration = end - start
    
    # Calculate 20% of the duration
    twenty_percent_duration = 0.2 * duration
    
    # Extract the segment of the audio corresponding to the svara
    y_svara = y[int(start * sr):int(end * sr)]
    
    # Extract the segment of the audio for the previous 20%
    prev_start = max(0, start - twenty_percent_duration)
    y_prev = y[int(prev_start * sr):int(start * sr)]
    
    # Extract the segment of the audio for the next 20%
    post_end = min(len(y) / sr, end + twenty_percent_duration)
    y_post = y[int(end * sr):int(post_end * sr)]
    
    # Extract features for the current svara, previous segment, and next segment
    svara_features = extract_features(y_svara, sr)
    prev_features = extract_features(y_prev, sr)
    post_features = extract_features(y_post, sr)
    
    # Append the features to the dataframe
    df_features = df_features.append({
        'id': id,
        'max pitch': svara_features[0], 'max pitch prev': prev_features[0], 'max pitch post': post_features[0],
        'min pitch': svara_features[1], 'min pitch prev': prev_features[1], 'min pitch post': post_features[1],
        'mean pitch': svara_features[2], 'mean pitch prev': prev_features[2], 'mean pitch post': post_features[2],
        'num change points': svara_features[3], 'num change points prev': prev_features[3], 'num change points post': post_features[3],
        'standard deviation': svara_features[4], 'standard deviation prev': prev_features[4], 'standard deviation post': post_features[4],
        'zero cross': svara_features[5], 'zero cross prev': prev_features[5], 'zero cross post': post_features[5],
        'loudness': svara_features[6], 'loudness prev': prev_features[6], 'loudness post': post_features[6],
        'amplitude envelope': svara_features[7], 'amplitude envelope prev': prev_features[7], 'amplitude envelope post': post_features[7],
        'band energy ratio': svara_features[8], 'band energy ratio prev': prev_features[8], 'band energy ratio post': post_features[8],
        'spectral centroid': svara_features[9], 'spectral centroid prev': prev_features[9], 'spectral centroid post': post_features[9],
        'spectral bandwidth': svara_features[10], 'spectral bandwidth prev': prev_features[10], 'spectral bandwidth post': post_features[10],
        'mfcc1': svara_features[11], 'mfcc1 prev': prev_features[11], 'mfcc1 post': post_features[11], 
        'mfcc2': svara_features[12], 'mfcc2 prev': prev_features[12], 'mfcc2 post': post_features[12],
        'mfcc3': svara_features[13], 'mfcc3 prev': prev_features[13], 'mfcc3 post': post_features[13],
        'mfcc4': svara_features[14], 'mfcc4 prev': prev_features[14], 'mfcc4 post': post_features[14],
        'mfcc5': svara_features[15], 'mfcc5 prev': prev_features[15], 'mfcc5 post': post_features[15],
        'mfcc6': svara_features[16], 'mfcc6 prev': prev_features[16], 'mfcc6 post': post_features[16],
        'mfcc7': svara_features[17], 'mfcc7 prev': prev_features[17], 'mfcc7 post': post_features[17],
        'mfcc8': svara_features[18], 'mfcc8 prev': prev_features[18], 'mfcc8 post': post_features[18],
        'mfcc9': svara_features[19], 'mfcc9 prev': prev_features[19], 'mfcc9 post': post_features[19],
        'mfcc10': svara_features[20], 'mfcc10 prev': prev_features[20], 'mfcc10 post': post_features[20],
        'mfcc11': svara_features[21], 'mfcc11 prev': prev_features[21], 'mfcc11 post': post_features[21],
        'mfcc12': svara_features[22], 'mfcc12 prev': prev_features[22], 'mfcc12 post': post_features[22],
        'mfcc13': svara_features[23], 'mfcc13 prev': prev_features[23], 'mfcc13 post': post_features[23],
        'is_sa': 1 if svara_type == 'sa' else 0,
        'is_ri': 1 if svara_type == 'ri' else 0,
        'is_pa': 1 if svara_type == 'pa' else 0,
        'is_ni': 1 if svara_type == 'ni' else 0,
        'is_ma': 1 if svara_type == 'ma' else 0,
        'is_ga': 1 if svara_type == 'ga' else 0,
        'is_dha': 1 if svara_type == 'dha' else 0
    }, ignore_index=True) 
    
    id += 1

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
df_normalized.to_csv('svara_features/context_features_normalized.csv', index=False)