import json
import langcodes as lc
import pandas as pd

all_langs = list(set(pd.read_csv('./data/avspeech_test_langs.csv')['Language']))
all_langs = [x for x in all_langs if type(x) == str]
print(all_langs)
lang_data = {}
for lang in all_langs:
    scores = []
    for other_lang in all_langs:
        scores.append(lc.tag_distance(lang, other_lang))
    lang_data[lang] = scores

json.dump(lang_data, open('lang_embeddings.json', 'w'))
