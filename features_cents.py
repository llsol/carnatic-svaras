import librosa
import numpy as np
import pandas as pd
import os
import compiam
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

headers = ['id',
           'max pitch (Cents)', 
           'min pitch (Cents)', 
           'mean pitch (Cents)', 
           'num change points', 
           'standard deviation (Cents)',
           'zero cross',
           'loudness',
           'amplitude envelope',
           'band energy ratio',
           'spectral centroid (Cents)',
           'spectral bandwidth (Cents)',
           'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5',
           'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10',
           'mfcc11', 'mfcc12', 'mfcc13',
           'is_sa',
           'is_ri',
           'is_pa',
           'is_ni',
           'is_ma',
           'is_ga',
           'is_dha']

# Create a dataframe
df_features = pd.DataFrame(columns=headers)

# Accessing the audio files
single_svara_path = os.path.join('single_svara')
id = 0

# Reference frequency in Hz
reference_frequency = 261.63  # C4 (Do central)

def hz_to_cents(hz, reference_frequency):
    return 1200 * np.log2(hz / reference_frequency)

# Iterate through the audio files to extract the current state-of-the-art selected pitch features
for file in os.listdir(single_svara_path):
    # Load the audio file
    y, sr = librosa.load(os.path.join(single_svara_path, file), mono=True)
    # Extract the pitch
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

    # Filter the valid frequencies
    valid_pitches = pitches[pitches > 0]

    # PITCH CURVE FEATURES
    # Max and Min Pitch
    max_pitch = np.max(valid_pitches)
    min_pitch = np.min(valid_pitches)

    # Convert to Cents
    max_pitch_cents = hz_to_cents(max_pitch, reference_frequency)
    min_pitch_cents = hz_to_cents(min_pitch, reference_frequency)

    # Mean Pitch and Standard Deviation values
    mean_pitch = np.mean(valid_pitches)
    std = np.std(valid_pitches)

    # Convert to Cents
    mean_pitch_cents = hz_to_cents(mean_pitch, reference_frequency)
    std_cents = hz_to_cents(std, reference_frequency)

    # Number of Change Points
    second_derivative = np.diff(np.diff(magnitudes))
    num_change_points = np.sum(second_derivative[:-1] * second_derivative[1:] < 0)

    # TIME DOMAIN FEATURES:
    # Mean Amplitude Envelope
    splits = np.array_split(y, 200)
    max_values = [np.max(np.abs(split)) for split in splits]
    mean_amplitude_envelope = np.mean(max_values)

    # Root Mean Square (Loudness)
    loudness = np.mean(librosa.feature.rms(y=y)[0])

    # Zero Crossing Rate
    zero_cross = np.mean(librosa.feature.zero_crossing_rate(y)[0])

    # FREQUENCY DOMAIN FEATURES:
    # Band Energy Ratio (BER)
    stft = np.abs(librosa.stft(y))
    ber = np.sum(stft[:stft.shape[0] // 2, :], axis=0) / np.sum(stft[stft.shape[0] // 2:, :], axis=0)
    band_energy_ratio = np.mean(ber)

    # Spectral Centroid
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_centroid_cents = hz_to_cents(spectral_centroid, reference_frequency)

    # Spectral Bandwidth
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    spectral_bandwidth_cents = hz_to_cents(spectral_bandwidth, reference_frequency)

    # Mel-Frequency Cepstral Coefficients (MFCCs)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)

    # Classify svara type
    svara_type = file.split('_')[0]

    # Extract the id
    id = int(file.split('_')[1].split('.')[0])

    # Append the features to the dataframe
    df_features = df_features.append({'id': id,
                                      'max pitch (Cents)': max_pitch_cents, 
                                      'min pitch (Cents)': min_pitch_cents, 
                                      'mean pitch (Cents)': mean_pitch_cents, 
                                      'num change points': num_change_points,
                                      'standard deviation (Cents)': std_cents,
                                      'zero cross': zero_cross,
                                      'loudness': loudness,
                                      'amplitude envelope': mean_amplitude_envelope,
                                      'band energy ratio': band_energy_ratio,
                                      'spectral centroid (Cents)': spectral_centroid_cents,
                                      'spectral bandwidth (Cents)': spectral_bandwidth_cents,
                                      'mfcc1': mfccs_mean[0], 'mfcc2': mfccs_mean[1], 'mfcc3': mfccs_mean[2],
                                      'mfcc4': mfccs_mean[3], 'mfcc5': mfccs_mean[4], 'mfcc6': mfccs_mean[5],
                                      'mfcc7': mfccs_mean[6], 'mfcc8': mfccs_mean[7], 'mfcc9': mfccs_mean[8],
                                      'mfcc10': mfccs_mean[9], 'mfcc11': mfccs_mean[10], 'mfcc12': mfccs_mean[11],
                                      'mfcc13': mfccs_mean[12],
                                      'is_sa': 1 if svara_type == 'sa' else 0,
                                      'is_ri': 1 if svara_type == 'ri' else 0,
                                      'is_pa': 1 if svara_type == 'pa' else 0,
                                      'is_ni': 1 if svara_type == 'ni' else 0,
                                      'is_ma': 1 if svara_type == 'ma' else 0,
                                      'is_ga': 1 if svara_type == 'ga' else 0,
                                      'is_dha': 1 if svara_type == 'dha' else 0,
                                      }, ignore_index=True)

# Save the dataframe to a CSV file
df_features.sort_values(by='id', inplace=True)
df_features.to_csv('features/features.csv', index=False)