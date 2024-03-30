import pandas as pd
import numpy as np
import langdetect
import yt_dlp



data = pd.read_csv('avspeech_test.csv')
data = np.array(data)[5:25]


for row in data:
    # command = f'yt-dlp -f "(bestvideo+bestaudio/best)[protocolup=dash]" --external-downloader ffmpeg --external-downloader-args "ffmpeg_i:-ss {row[1]} -to {row[2]}" --format 18  "https://www.youtube.com/watch?v={row[0]}"'
    # #subprocess.run(shlex.split(command))

    # try:
    # # Object creation using YouTube
    # # which was imported in the beginning
    #     yt = YouTube(f"https://www.youtube.com/watch?v={row[0]}")
    # except:
    #     # Handle exception
    #     print("Connection Error")

    # # Get all streams and filter for mp4 files
    # title = yt.title
    # print(title)
    # 
    # print(lang)

    ydl_opts = {
        'format': '18',
        'download-sections': "*1-10"
        # 'downloader': 'ffmpeg',
        # 'downloader-args': ['ffmpeg_i', '-ss {row[1]} -to {row[2]}']
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download(f'https://www.youtube.com/watch?v={row[0]}')
        info_dict = ydl.extract_info(f'https://www.youtube.com/watch?v={row[0]}')
        title = info_dict.get('title')
    print(title)
    lang = langdetect.detect(title)
    print(lang)
    break

