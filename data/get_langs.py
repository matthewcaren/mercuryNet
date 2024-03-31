import langdetect
import pandas as pd
from pytube import YouTube

data = pd.read_csv('avspeech_test.csv')
langs = []
for row in data.iterrows():
    if row[0] % 1000 == 0:
        print(row[0])
    lang = None
    try:
        yt = YouTube(f'https://www.youtube.com/watch?v={row[1][0]}')
        title = yt.title
        lang = langdetect.detect(title)          
    except:
        pass
        #print("Not available")
    langs.append(lang)
data.insert(len(data.columns), 'Language', langs)
data.to_csv("avspeech_test_langs.csv")