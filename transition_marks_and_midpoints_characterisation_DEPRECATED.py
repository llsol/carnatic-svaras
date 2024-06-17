import librosa
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

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

# Accessing the audio file
audio_file = os.path.join('svara_task', 'separated_data', 'voice_separated.mp3')

# Load audio file
y, sr = librosa.load(audio_file, mono=True)

# Define window size (e.g., 0.1 seconds) for segments before and after transition
window_size = 0.1  # seconds

# Function to extract features from an audio segment
def extract_features(segment):
    features = {}
    
    # Extract the pitch
    pitches, magnitudes = librosa.piptrack(y=segment, sr=sr)
    valid_pitches = pitches[pitches > 0]

    if len(valid_pitches) > 0:
        # PITCH CURVE FEATURES
        second_derivative = np.diff(np.diff(magnitudes))
        features['num_change_points'] = np.sum(second_derivative[:-1] * second_derivative[1:] < 0)
        features['std'] = np.std(valid_pitches)
    else:
        features['num_change_points'] = np.nan
        features['std'] = np.nan
        
    # TIME DOMAIN FEATURES:
    splits = np.array_split(segment, 200)
    max_values = [np.max(np.abs(split)) for split in splits]
    features['mean_amplitude_envelope'] = np.mean(max_values)
    features['loudness'] = np.mean(librosa.feature.rms(y=segment)[0])
    features['zero_cross'] = np.mean(librosa.feature.zero_crossing_rate(y=segment)[0])

    # FREQUENCY DOMAIN FEATURES:
    stft = np.abs(librosa.stft(segment))
    epsilon = 1e-10  # Small value to avoid division by zero
    features['ber'] = np.sum(stft[:stft.shape[0] // 2, :], axis=0) / (np.sum(stft[stft.shape[0] // 2:, :], axis=0) + epsilon)
    features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=segment, sr=sr))
    features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(y=segment, sr=sr))
    
    return features

# Initialize lists to store the feature differences for transitions and non-transitions
feature_diffs_transitions = []
feature_diffs_non_transitions = []

# Identify transition points
transition_times = annotations['start'].tolist()

# Calculate midpoints between transition points
midpoints = [(transition_times[i] + transition_times[i+1]) / 2 for i in range(len(transition_times) - 1)]

# Function to process a list of times (either transitions or midpoints)
def process_times(times, label):
    feature_diffs = []
    for time_point in times:
        # Segment before and after the point
        before_segment = y[int((time_point - window_size) * sr):int(time_point * sr)]
        after_segment = y[int(time_point * sr):int((time_point + window_size) * sr)]
        
        # Extract features from both segments
        features_before = extract_features(before_segment)
        features_after = extract_features(after_segment)
        
        # Compute the differences in features
        if features_before and features_after:  # Ensure both feature sets are not empty
            feature_diff = {f'{key}_diff': features_after[key] - features_before[key] for key in features_before}
            feature_diff['label'] = label
            feature_diffs.append(feature_diff)
    
    return feature_diffs

# Process transition points
feature_diffs_transitions = process_times(transition_times, label='transition')

# Process midpoints as non-transition points
feature_diffs_non_transitions = process_times(midpoints, label='non_transition')

# Combine the feature differences
all_feature_diffs = feature_diffs_transitions + feature_diffs_non_transitions

# Create a DataFrame from the combined feature differences
df_feature_diffs = pd.DataFrame(all_feature_diffs)

# Print the feature differences DataFrame
print(df_feature_diffs)

# Save the feature differences to a CSV file
if not os.path.exists('svara_transition'):
    os.makedirs('svara_transition')

df_feature_diffs.to_csv('svara_transition/feature_differences_with_labels.csv', index=False)
