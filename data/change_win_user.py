import numpy as np

PATH = 'english_windows.npy'
prev_user = 'jrajagop'
new_user = 'mcaren'

original = np.load(PATH, allow_pickle=True)

new_list = []
for e in original:
    original_path = e[0]
    new_path = original_path.replace(prev_user, new_user)
    new_list.append(new_path)

np.save(PATH, np.array(new_list), allow_pickle=True)