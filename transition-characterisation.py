import librosa
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

# Save annotation columns and read annotations file 
annotation_columns = ['type', 'start', 'end', 'duration', 'svara']
annotations_path = os.path.join('svara_task', 'kamakshi_new.txt')
annotations = pd.read_csv(annotations_path, sep='\t', names=annotation_columns)

# Convert 'start' and 'end' columns to seconds
def time_to_seconds(time_str):
    h, m, s = map(float, time_str.split(':'))
    return h * 3600 + m * 60 + s

annotations['start'] = annotations['start'].apply(time_to_seconds)
annotations['end'] = annotations['end'].apply(time_to_seconds)

# Define headers for the dataframe
headers = [
    'id', 'num change points', 'standard deviation',
    'zero cross', 'loudness', 'amplitude envelope', 'band energy ratio', 'spectral centroid',
    'spectral bandwidth', 'pitch stability', 'is_transition'
]

# Create a dataframe
df_svara_transition = pd.DataFrame(columns=headers)

# Accessing the audio file
audio_file = os.path.join('svara_task', 'separated_data', 'voice_separated.mp3')

# Window size in seconds
window_size = 0.1

# Load audio file
y, sr = librosa.load(audio_file, mono=True)

# Calculate audio duration
audio_duration = librosa.get_duration(y=y, sr=sr)

# Create windows of 0.1 seconds
windows = np.arange(0, audio_duration, window_size)

def compute_pitch_stability(y, sr, hop_length=512):
    # Extract the pitch contour
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr, hop_length=hop_length)
    
    # Filter out non-zero pitch values
    pitch_values = pitches[pitches > 0]
    
    # Check if there are enough pitch values to compute stability
    if len(pitch_values) == 0:
        return np.nan  # or some other value indicating undefined stability
    
    # Compute the standard deviation of the pitch values
    pitch_std = np.std(pitch_values)
    
    # Define pitch stability (low std indicates high stability)
    pitch_stability = 1 / (pitch_std + 1e-10)  # Add a small epsilon to avoid division by zero
    
    return pitch_stability

id = 0

for start_time in windows:
    end_time = start_time + window_size
    segment = y[int(start_time * sr):int(end_time * sr)]

    # Extract the pitch
    pitches, magnitudes = librosa.piptrack(y=segment, sr=sr)
    valid_pitches = pitches[pitches > 0]

    if len(valid_pitches) == 0:
        continue

    # PITCH CURVE FEATURES
    second_derivative = np.diff(np.diff(magnitudes))
    num_change_points = np.sum(second_derivative[:-1] * second_derivative[1:] < 0)
    std = np.std(valid_pitches)

    # TIME DOMAIN FEATURES:
    splits = np.array_split(segment, 200)
    max_values = [np.max(np.abs(split)) for split in splits]
    mean_amplitude_envelope = np.mean(max_values)
    loudness = np.mean(librosa.feature.rms(y=segment)[0])
    zero_cross = np.mean(librosa.feature.zero_crossing_rate(y=segment)[0])

    # FREQUENCY DOMAIN FEATURES:
    stft = np.abs(librosa.stft(segment))
    epsilon = 1e-10  # Small value to avoid division by zero
    ber = np.sum(stft[:stft.shape[0] // 2, :], axis=0) / (np.sum(stft[stft.shape[0] // 2:, :], axis=0) + epsilon)
    band_energy_ratio = np.mean(ber[~np.isnan(ber)])  # Ignore NaN values

    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=segment, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=segment, sr=sr))

    # Compute pitch stability
    pitch_stability = compute_pitch_stability(segment, sr)

    # Check if there is a transition between svaras in the current window
    is_transition = 0
    for index, row in annotations.iterrows():
        if (row['start'] >= start_time and row['start'] < end_time) or (row['end'] >= start_time and row['end'] < end_time):
            is_transition = 1
            break

    # Append the features to the dataframe
    new_row = {
        'id': id,
        'num change points': num_change_points, 'standard deviation': std, 'zero cross': zero_cross,
        'loudness': loudness, 'amplitude envelope': mean_amplitude_envelope, 'band energy ratio': band_energy_ratio,
        'spectral centroid': spectral_centroid, 'spectral bandwidth': spectral_bandwidth,
        'pitch stability': pitch_stability, 'is_transition': int(is_transition)
    }

    df_svara_transition = pd.concat([df_svara_transition, pd.DataFrame([new_row], columns=headers)], ignore_index=True)
    id += 1

# Sort dataframe by id to ensure sequential order
df_svara_transition.sort_values(by='id', inplace=True)

# Normalization using Z-score
scaler = StandardScaler()

# Separate the 'id' and transition label columns from the features to be normalized
ids = df_svara_transition['id']
is_transition = df_svara_transition['is_transition']
features_to_normalize = df_svara_transition.drop(columns=['id', 'is_transition'])

# Apply Z-score normalization
normalized_features = scaler.fit_transform(features_to_normalize)

# Recreate the DataFrame with normalized features
df_tf_normalized = pd.DataFrame(normalized_features, columns=features_to_normalize.columns)

# Add back the 'id' and transition label columns
df_tf_normalized['id'] = ids
df_tf_normalized['is_transition'] = is_transition

# Ensure the 'id' column is the first column
df_tf_normalized = df_tf_normalized[['id'] + features_to_normalize.columns.tolist() + ['is_transition']]

# Save the normalized dataframe to a CSV file
if not os.path.exists('svara_transition'):
    os.makedirs('svara_transition')

df_tf_normalized.to_csv('svara_transition/svara_transitions_normalized.csv', index=False)
