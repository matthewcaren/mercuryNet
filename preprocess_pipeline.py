import pandas as pd
import numpy as np
import subprocess
import shlex
import time
import os
import json
import cv2
from ultralytics import YOLO
import librosa
import shutil 

AUDIO_SR = 22050
VID_FRAME_RATE = 30
HOP_SIZE = AUDIO_SR // VID_FRAME_RATE
assert((22050/VID_FRAME_RATE) % 1 == 0)

def extract_features(wav_path):
    '''
    extract f0 and voicedness from centered windows
    '''
    samples, sr = librosa.load(wav_path)
    f0, voiced_flag, voiced_probs = librosa.pyin(samples,
                                                 sr = sr,
                                                 fmin = 50,
                                                 fmax = 2500,
                                                 frame_length = HOP_SIZE*4,
                                                 win_length = HOP_SIZE*2,
                                                 hop_length = HOP_SIZE,
                                                 pad_mode = 'constant',
                                                 center=True)
    rms = librosa.feature.rms(y = samples,
                              frame_length = HOP_SIZE*4,
                              hop_length = HOP_SIZE,
                              pad_mode='constant',
                              center=True).squeeze()
    return np.stack((f0, voiced_flag, rms))

def process_rows(dataFrame, verbose=False):
    model = YOLO('yolov8m-face.pt')
    batch_size = 32
    output_dir = './vids'
    for row in dataFrame:
        # Download video and make directories
        vid_id = f"{row[1]}_{str(row[2])}"
        vid_dir = os.path.join(output_dir, vid_id)
        if not os.path.exists(vid_dir):
            os.mkdir(vid_dir)
        vid_path = os.path.join(output_dir, vid_id, f'{vid_id}.mp4')
        wav_path = os.path.join(output_dir, vid_id, f'{vid_id}.wav')
        if not os.path.exists(vid_path):
            command = f'yt-dlp --extractor-args "youtube:player_client=web" -q -f "(bestvideo+bestaudio/best)[protocolup=dash]" -o {vid_id}.mp4 --download-sections *{row[2]}-{row[3]} --format 18 -P {vid_dir}  "https://www.youtube.com/watch?v={vid_id}"'
            subprocess.run(shlex.split(command))
        if not os.path.exists(vid_path):
            os.rmdir(vid_dir)
        else:
            # Extract audio and facial features (nill for now)
            if not os.path.exists(wav_path):
                subprocess.run(['ffmpeg', '-i', vid_path, '-ac', '1',  wav_path,'-loglevel', 'warning'])
            facial_attributes = {'race': 0, 'gender': 0, 'emotion': 0, 'age': 0, 'lang': row[6]}
            json.dump(facial_attributes, open(f'{vid_dir}/{vid_id}_feat.json', 'w'))
            
            # Extract frames from video
            frames = []
            video_stream = cv2.VideoCapture(vid_path)
            while True:
                still_reading, frame = video_stream.read()
                if not still_reading:
                    video_stream.release()
                    break
                frames.append(frame)
            [x, y, _] = frames[0].shape

            # Make ground truth data, only take as many frames as we have
            ground_truth = extract_features(wav_path)[:, :len(frames)]
            np.save(f'{vid_dir}/{vid_id}_pros.npy', ground_truth)

            # Run YOLO on all frames
            head_center = np.array([row[4]*y, row[5]*x])
            batches = [frames[i:i + batch_size] for i in range(0, len(frames), batch_size)]
            counter, failed_detection = 0, False
            for batch in batches:
                results = model.predict(batch, device='mps', verbose=False)
                for result in results:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    if len(boxes) == 0:
                        failed_detection = True
                    elif len(boxes) == 1:
                        correct_box = boxes[0]
                    else:
                        # Ensure center of frame is close enough to annotated person
                        centers = [np.array([0.5*(box[2] + box[0]), 0.5*(box[3] + box[1])]) for box in boxes]
                        dists = [np.linalg.norm(center - head_center) for center in centers]
                        if (min(dists) < 20) or (dists[np.argsort(dists)[1]] > 3*min(dists)):
                            correct_box = boxes[np.argmin(dists)]
                        else:
                            failed_detection = True
                    x1, y1, x2, y2 = [round(x) for x in correct_box]
                    cv2.imwrite(f'{vid_dir}/{vid_id}_{counter}.jpg', result.orig_img[y1:y2,x1:x2, :])
                    counter += 1
            if failed_detection:
                if verbose: print("Could not confidently detect faces in all frames for", vid_dir)
                shutil.rmtree(vid_dir)
                
if __name__ == '__main__':
    start = time.time()
    raw_data = pd.read_csv('./data/avspeech_test_langs.csv')
    dataFrame = np.array(raw_data)[800:810]
    process_rows(dataFrame)
    print(time.time() - start)