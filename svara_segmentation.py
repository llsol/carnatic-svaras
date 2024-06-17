import os
import librosa
import numpy as np
import pandas as pd
from scipy.io import wavfile

def procesar_audio(audio_path, svara_path, output_dir):
    # Cargar el archivo de audio
    y, sr = librosa.load(audio_path, sr=None)
    
    # Cargar el archivo de texto como un DataFrame con pandas
    headers = ['Level', 'Start', 'End', 'Duration', 'Label']
    svara = pd.read_csv(svara_path, sep='\t', names=headers)
    
    # Modificar el DataFrame svara
    svara.insert(0, 'ID', range(1, len(svara) + 1))
    svara.set_index('ID', inplace=True)
    
    for index, row in svara.iterrows():
        # Convertir tiempos de inicio y fin de string a float
        start_time = row['Start']
        end_time = row['End']
        duration = row['Duration']
        
        hours_s, minutes_s, seconds_s = map(float, start_time.split(':'))
        seconds_s += hours_s * 3600 + minutes_s * 60

        hours_e, minutes_e, seconds_e = map(float, end_time.split(':'))
        seconds_e += hours_e * 3600 + minutes_e * 60

        hours_d, minutes_d, seconds_d = map(float, duration.split(':'))
        seconds_d += hours_d * 3600 + minutes_d * 60

        # Actualizar el DataFrame con los valores modificados
        svara.at[index, 'Start'] = seconds_s
        svara.at[index, 'End'] = seconds_e
        svara.at[index, 'Duration'] = seconds_d
    
    svara_list = ['sa', 'ri', 'ga', 'ma', 'pa', 'dha', 'ni']
    
    # Filtrar el DataFrame por cada svara en svara_list
    for element in svara_list:
        filtered_df = svara[svara['Label'] == element]
        globals()['svara_' + element] = filtered_df
        
        # Crear una carpeta 'single_svara' si no existe para almacenar los segmentos de audio
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Extraer y guardar los segmentos de audio correspondientes
        for index, row in globals()['svara_' + element].iterrows():
            start_sample = int(row['Start'] * sr)
            end_sample = int(row['End'] * sr)
            y_segment = y[start_sample:end_sample]
            
            audio_segment_path = os.path.join(output_dir, element + '_' + str(index) + '.wav')
            wavfile.write(audio_segment_path, sr, y_segment)