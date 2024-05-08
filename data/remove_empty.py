import numpy as np
from tqdm import tqdm
windows = np.load('data/windows_english.npy', allow_pickle=True)
new_windows = []
print(len(windows))
for window, [s, e] in tqdm(windows):
    pros_path = window+'_pros.npy'
    pros = np.load(pros_path)[1, s:e]
    if np.mean(pros) > 0.1:
        new_windows.append([window, [s, e]])
print(len(new_windows))
np.save('data/filtered_en.npy', new_windows)
        