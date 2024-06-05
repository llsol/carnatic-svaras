import pandas as pd
import os

svara_path = os.path.join('svara_task', 'kamakshi_new.txt')
data = pd.read_csv(svara_path, delimiter='\t', names=['type', 'start', 'end', 'duration', 'svara'])

svara_data = data[data['type'].str.strip() == 'svara']

svara_data['start'] = pd.to_timedelta(svara_data['start'].str.strip())
svara_data['end'] = pd.to_timedelta(svara_data['end'].str.strip())

svara_data['next_start'] = svara_data['start'].shift(-1)
svara_data['silence_duration'] = (svara_data['next_start'] - svara_data['end']).fillna(pd.Timedelta(seconds=0))
silence_rows = svara_data[svara_data['silence_duration'] > pd.Timedelta(seconds=0)]

silence_data = pd.DataFrame({
    'start': silence_rows['end'],
    'end': silence_rows['next_start'],
    'svara': 'silence'
})

all_data = pd.concat([svara_data[['start', 'end', 'svara']], silence_data]).sort_values(by='start').reset_index(drop=True)

svaras = ['sa', 'ri', 'ga', 'ma', 'pa', 'dha', 'ni', 'silence']

transition_matrix = pd.DataFrame(0, index=svaras, columns=svaras, dtype=float)

for i in range(len(all_data) - 1):
    current_svara = all_data.loc[i, 'svara']
    next_svara = all_data.loc[i + 1, 'svara']
    transition_matrix.loc[current_svara, next_svara] += 1

transition_matrix = transition_matrix.div(transition_matrix.sum(axis=1), axis=0).fillna(0)

print(transition_matrix)