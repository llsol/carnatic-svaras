import librosa
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

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
def extract_features(segment, previous_or_next):
    features = {}

    # Extract the pitch
    pitches, magnitudes = librosa.piptrack(y=segment, sr=sr)
    valid_pitches = pitches[pitches > 0]

    if len(valid_pitches) > 0:
        # PITCH CURVE FEATURES
        second_derivative = np.diff(np.diff(magnitudes))
        features[f'num_change_points_{previous_or_next}'] = np.sum(second_derivative[:-1] * second_derivative[1:] < 0)
        features[f'std_{previous_or_next}'] = np.std(valid_pitches)
    else:
        features[f'num_change_points_{previous_or_next}'] = np.nan
        features[f'std_{previous_or_next}'] = np.nan

    # TIME DOMAIN FEATURES:
    splits = np.array_split(segment, 200)
    max_values = [np.max(np.abs(split)) for split in splits]
    features[f'mean_amplitude_envelope_{previous_or_next}'] = np.mean(max_values)
    features[f'loudness_{previous_or_next}'] = np.mean(librosa.feature.rms(y=segment)[0])
    features[f'zero_cross_{previous_or_next}'] = np.mean(librosa.feature.zero_crossing_rate(y=segment)[0])

    # FREQUENCY DOMAIN FEATURES:
    stft = np.abs(librosa.stft(segment))
    epsilon = 1e-10  # Small value to avoid division by zero
    features[f'ber_{previous_or_next}'] = np.sum(stft[:stft.shape[0] // 2, :], axis=0) / (np.sum(stft[stft.shape[0] // 2:, :], axis=0) + epsilon)
    features[f'spectral_centroid_{previous_or_next}'] = np.mean(librosa.feature.spectral_centroid(y=segment, sr=sr))
    features[f'spectral_bandwidth_{previous_or_next}'] = np.mean(librosa.feature.spectral_bandwidth(y=segment, sr=sr))

    return features

# Initialize lists to store the feature differences for transitions
feature_transitions = []

# Identify transition points
transition_times = annotations['start'].tolist()

# Calculate midpoints between transition points
midpoints = [(transition_times[i] + transition_times[i+1]) / 2 for i in range(len(transition_times) - 1)]

# Function to process a list of times (either transitions or midpoints)
def process_times(times, label):
    features = []
    for time_point in times:
        # Segment before and after the point
        before_segment = y[int((time_point - window_size) * sr):int(time_point * sr)]
        after_segment = y[int(time_point * sr):int((time_point + window_size) * sr)]

        # Extract features from both segments
        features_before = extract_features(before_segment, previous_or_next='prev')
        features_after = extract_features(after_segment, previous_or_next='next')

        # Append the features to the list
        features.append(features_before)
        features.append(features_after)
        if label == 'transition':
            features[-1]['label'] = 1

    return features

# Process transition points
feature_transitions = process_times(transition_times, label='transition')

# Create a DataFrame from the combined feature differences
feature_transitions = pd.DataFrame(feature_transitions)

# Print the feature differences DataFrame
print(feature_transitions)

# Save the feature differences to a CSV file
if not os.path.exists('svara_transition'):
    os.makedirs('svara_transition')

feature_transitions.to_csv('svara_transition/svara_transitions_method_2.csv', index=False)
