import pandas as pd
import numpy as np

all_windows = np.load('data/all_windows.npy', allow_pickle=True)
all_data = pd.read_csv('data/avspeech_test_langs.csv')
all_data = all_data[all_data['Language'] == 'en']
allowed_ids = set()
for _, row in all_data.iterrows():
    allowed_ids.add(row['ID'] + '_' + str(row['s']))
new_windows = [window for window in all_windows if window[0].split('/')[-1] in allowed_ids]
np.save('data/windows_english.npy', new_windows)


[32, 90, 3]