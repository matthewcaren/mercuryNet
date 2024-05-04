import pandas as pd

data = pd.read_csv('data/avspeech_test_langs.csv')
data.to_csv('data/avspeech_test_langs.csv', index=False)