import numpy as np

PATH = 'english_windows.npy'
prev_user = 'jrajagop'
new_user = 'mcaren'

original = np.load(PATH, allow_pickle=True)

new_list = []
for p, [s, e] in original:
    original_path = p
    new_path = [original_path.replace(prev_user, new_user), [s, e]]
    new_list.append(new_path)

np.save(PATH, np.array(new_list), allow_pickle=True)