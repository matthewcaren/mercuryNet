import pandas as pd

data = pd.read_csv('avspeed_test_langs.csv')
all_langs = set(data['Language'])
print(data['Language'].value_counts())
data = data[(data['Language'].isnull())]
print(len(data))
