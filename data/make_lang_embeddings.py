import json
import langcodes as lc
import pandas as pd

all_langs = list(set(pd.read_csv('/data/avspeech_test_langs.csv')['Language']))
all_langs = [x for x in all_langs if type(x) == str]
lang_data = {}
for lang in all_langs:
    scores = []
    for other_lang in all_langs:
        if lang == 'nolang' or other_lang == 'nolang':
            scores.append(200)
        else:
            scores.append(lc.tag_distance(lang, other_lang))
    scores = [s/134 for s in scores]
    lang_data[lang] = scores
json.dump(lang_data, open('data/lang_embeddings.json', 'w'))
