import numpy as np
import os
def make_all_windows(root_dir, directories, overlap=30, window_size=90):        
    windows = []
    overlap = overlap
    window_size = window_size
    for vid_dir in directories:
        if vid_dir[0] != '.':
            frames = np.load(f'{root_dir}/{vid_dir}/{vid_dir}_frames.npy')
            num_frames = frames.shape[0]
            vid_windows = get_windows(overlap, num_frames, window_size)
            for window in vid_windows:
                windows.append((f'{root_dir}/{vid_dir}/{vid_dir}', window))
    np.save('data/all_windows.npy', windows)
                
def get_windows(overlap, num_images, window_size):
    num_windows = (num_images - overlap) // (window_size - overlap)
    num_frames_in_window = num_windows*(window_size - overlap) + overlap
    amount_to_chop_front = (num_images - num_frames_in_window) // 2
    windows = []
    for i in range(num_windows):
        start = i*(window_size - overlap) + amount_to_chop_front
        windows.append([start, start + window_size])
    return windows

make_all_windows('/nobackup/users/jrajagop/vids', os.listdir('/nobackup/users/jrajagop/vids'))