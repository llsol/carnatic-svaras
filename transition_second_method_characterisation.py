import librosa
import numpy as np
import pandas as pd
import os

# Function to convert time string to seconds
def time_to_seconds(time_str):
    h, m, s = map(float, time_str.split(':'))
    return h * 3600 + m * 60 + s

# Function to read segmentation data from a .txt file without headers
def read_segmentation_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            data.append(parts)
    return data

# Function to extract features from an audio segment
def extract_features(y_segment, sr):
    pitches, magnitudes = librosa.piptrack(y=y_segment, sr=sr)

    valid_pitches = pitches[pitches > 0]
    max_pitch = np.max(valid_pitches) if valid_pitches.size > 0 else 0
    min_pitch = np.min(valid_pitches) if valid_pitches.size > 0 else 0
    mean_pitch = np.mean(valid_pitches) if valid_pitches.size > 0 else 0
    std_pitch = np.std(valid_pitches) if valid_pitches.size > 0 else 0

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

    return [
        max_pitch, min_pitch, mean_pitch, num_change_points, std_pitch,
        zero_cross, loudness, mean_amplitude_envelope, band_energy_ratio,
        spectral_centroid, spectral_bandwidth,
        *mfccs_mean
    ]

# Function to create the feature dataset from fixed-length segments
def create_feature_dataset(y, sr, segment_length_sec, segmentation_data):
    segment_length_samples = int(segment_length_sec * sr)
    num_segments = len(y) // segment_length_samples
    feature_dataset = []

    for i in range(num_segments - 1):
        start1 = i * segment_length_samples
        end1 = start1 + segment_length_samples
        start2 = end1
        end2 = start2 + segment_length_samples

        y_segment1 = y[start1:end1]
        y_segment2 = y[start2:end2]

        features1 = extract_features(y_segment1, sr)
        features2 = extract_features(y_segment2, sr)

        segment_id = f"({i+1},{i+2})"
        is_transition = 0

        # Calculate times in seconds for the halves of the segments
        segment1_middle_time = (start1 + end1) / 2 / sr
        segment2_middle_time = (start2 + end2) / 2 / sr

        # Check for transition based on segmentation data
        for j in range(len(segmentation_data) - 1):
            end_time_svara_i = time_to_seconds(segmentation_data[j][3])
            start_time_svara_i_plus_1 = time_to_seconds(segmentation_data[j + 1][2])

            if (segment1_middle_time <= end_time_svara_i <= end1 / sr) or \
               (start2 / sr <= start_time_svara_i_plus_1 <= segment2_middle_time):
                is_transition = 1
                break

        feature_row = [segment_id] + features1 + features2 + [is_transition]
        feature_dataset.append(feature_row)

    return feature_dataset

# Define the headers for the new feature dataset
new_headers = [
    'id',
    'max_pitch_1', 'min_pitch_1', 'mean_pitch_1', 'num_change_points_1', 'std_pitch_1',
    'zero_cross_1', 'loudness_1', 'amplitude_envelope_1', 'band_energy_ratio_1',
    'spectral_centroid_1', 'spectral_bandwidth_1',
    'mfcc1_1', 'mfcc2_1', 'mfcc3_1', 'mfcc4_1', 'mfcc5_1',
    'mfcc6_1', 'mfcc7_1', 'mfcc8_1', 'mfcc9_1', 'mfcc10_1',
    'mfcc11_1', 'mfcc12_1', 'mfcc13_1',
    'max_pitch_2', 'min_pitch_2', 'mean_pitch_2', 'num_change_points_2', 'std_pitch_2',
    'zero_cross_2', 'loudness_2', 'amplitude_envelope_2', 'band_energy_ratio_2',
    'spectral_centroid_2', 'spectral_bandwidth_2',
    'mfcc1_2', 'mfcc2_2', 'mfcc3_2', 'mfcc4_2', 'mfcc5_2',
    'mfcc6_2', 'mfcc7_2', 'mfcc8_2', 'mfcc9_2', 'mfcc10_2',
    'mfcc11_2', 'mfcc12_2', 'mfcc13_2',
    'is_transition'
]

# Paths
input_file_path = 'svara_task/kamakshi_new.txt'  # Replace with your actual file path
audio_file_path = 'svara_task/Kamakshi/Sanjay Subrahmanyan - Kamakshi.mp3'  # Replace with your actual audio file path

# Read the segmentation data
segmentation_data = read_segmentation_data(input_file_path)

# Load the audio file
y, sr = librosa.load(audio_file_path, sr=None)

# Define the segment length in seconds
segment_length_sec = 0.1

# Create the feature dataset
feature_dataset = create_feature_dataset(y, sr, segment_length_sec, segmentation_data)

# Write the dataset to a .csv file
output_file_path = 'svara_transition/svara_transitions_method_2.csv'
df_feature_dataset = pd.DataFrame(feature_dataset, columns=new_headers)
df_feature_dataset.to_csv(output_file_path, index=False)

print("Feature dataset created and saved to 'svara_transitions_method_2.csv'")
