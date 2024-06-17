import os
import numpy as np
import pandas as pd
import librosa
from sklearn.preprocessing import StandardScaler

def characterisation_labels(audio_file, annotations_file, output_csv='labels_features.csv'):
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
    y, sr = librosa.load(audio_file, mono=True)

    # Read the annotations file
    annotation_columns = ['type', 'start', 'end', 'duration', 'svara']
    annotations = pd.read_csv(annotations_file, delim_whitespace=True, header=None, names=annotation_columns)

    # Convert 'start' and 'end' columns to seconds
    def time_to_seconds(time_str):
        h, m, s = map(float, time_str.split(':'))
        return h * 3600 + m * 60 + s

    annotations['start'] = annotations['start'].apply(time_to_seconds)
    annotations['end'] = annotations['end'].apply(time_to_seconds)

    # Reference frequency in Hz for pitch conversion to cents
    reference_frequency = 261.63  # C4 (Do central)

    # Convert Hz to cents
    def hz_to_cents(hz, reference_frequency):
        return 1200 * np.log2(hz / reference_frequency)

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
        
        # Handle infinite values
        if np.isinf(max_pitch) or np.isinf(min_pitch) or np.isinf(mean_pitch) or np.isinf(std):
            # Replace infinite values with NaN or a large number
            max_pitch_cents = np.nan
            min_pitch_cents = np.nan
            mean_pitch_cents = np.nan
            std_cents = np.nan
        else:
            max_pitch_cents = hz_to_cents(max_pitch, reference_frequency)
            min_pitch_cents = hz_to_cents(min_pitch, reference_frequency)
            mean_pitch_cents = hz_to_cents(mean_pitch, reference_frequency)
            std_cents = hz_to_cents(std, reference_frequency)

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
        
        return [max_pitch_cents, min_pitch_cents, mean_pitch_cents, num_change_points, std_cents, zero_cross, loudness, mean_amplitude_envelope, band_energy_ratio, spectral_centroid, spectral_bandwidth] + list(mfccs_mean)

    # Iterate through each row in the annotations file
    for _, row in annotations.iterrows():
        start = row['start']
        end = row['end']
        svara_type = row['svara']
        
        # Extract the segment of the audio corresponding to the svara
        y_svara = y[int(start * sr):int(end * sr)]
        
        # Extract features for the current svara
        svara_features = extract_features(y_svara, sr)
        
        # Append the features to the dataframe only if there are no infinite values
        if not any(np.isnan(f) for f in svara_features):
            df_features = df_features.append({
                'id': id,
                'max pitch': svara_features[0],
                'min pitch': svara_features[1],
                'mean pitch': svara_features[2],
                'num change points': svara_features[3],
                'standard deviation': svara_features[4],
                'zero cross': svara_features[5],
                'loudness': svara_features[6],
                'amplitude envelope': svara_features[7],
                'band energy ratio': svara_features[8],
                'spectral centroid': svara_features[9],
                'spectral bandwidth': svara_features[10],
                'mfcc1': svara_features[11], 'mfcc2': svara_features[12], 'mfcc3': svara_features[13],
                'mfcc4': svara_features[14], 'mfcc5': svara_features[15], 'mfcc6': svara_features[16],
                'mfcc7': svara_features[17], 'mfcc8': svara_features[18], 'mfcc9': svara_features[19],
                'mfcc10': svara_features[20], 'mfcc11': svara_features[21], 'mfcc12': svara_features[22],
                'mfcc13': svara_features[23],
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

    # Save the dataframe to a CSV file
    df_features.sort_values(by='id', inplace=True)
    output_csv = 'svara_features/labels_features.csv'
    df_features.to_csv(output_csv, index=False)

    return output_csv


