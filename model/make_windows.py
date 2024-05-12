import numpy as np
import os
from tqdm import tqdm
import pandas as pd

def make_all_windows(root_dir, allowed_directories, overlap=30, window_size=90):    
    """
    Makes the actual windows for list of videos
    """
    windows = []
    overlap = overlap
    window_size = window_size
    for vid_dir in tqdm(allowed_directories):
        frames = np.load(f'{root_dir}/{vid_dir}/{vid_dir}_frames.npy')
        num_frames = frames.shape[0]
        vid_windows = get_windows(overlap, num_frames, window_size)
        features = np.load(f'{root_dir}/{vid_dir}/{vid_dir}_pros.npy')
        freq, voiced = features[0, :], features[1, :]
        for window in vid_windows:
            freq_window = freq[window[0]:window[1]]
            voiced_window = voiced[window[0]:window[1]]
            # Check that prosody is not all 0 and is the right size
            if np.mean(voiced_window) > 0.1 and voiced_window.shape[0] == 90:
                voiced_0 = np.where(voiced_window == 0) 
                freq_nan = np.where(np.isnan(freq_window) == True)
                if np.array_equal(voiced_0, freq_nan):
                    windows.append((f'{vid_dir}/{vid_dir}', window[0], window[1]))
    return windows
                
def get_windows(overlap, num_images, window_size):
    num_windows = (num_images - overlap) // (window_size - overlap)
    num_frames_in_window = num_windows*(window_size - overlap) + overlap
    amount_to_chop_front = (num_images - num_frames_in_window) // 2
    windows = []
    for i in range(num_windows):
        start = i*(window_size - overlap) + amount_to_chop_front
        windows.append([start, start + window_size])
    return windows

def run_windows_english(root_dir, avspeech_loc, output_loc):
    """
    Makes and saves windows for all the english videos in a directory
    """
    all_data = pd.read_csv(avspeech_loc)
    all_data = all_data[all_data['Language'] == 'en']
    allowed_ids = set()
    for _, row in all_data.iterrows():
        allowed_ids.add(row['ID'] + '_' + str(row['s']))
    allowed_directories = [direc for direc in os.listdir(root_dir) if direc[0] != '.' and direc in allowed_ids]
    windows = make_all_windows(root_dir, allowed_directories)
    np.save(output_loc, windows)
