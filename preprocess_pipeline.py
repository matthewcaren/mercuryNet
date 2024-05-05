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
import argparse
from deepface import DeepFace

AUDIO_SR = 22050
VID_FRAME_RATE = 30
HOP_SIZE = AUDIO_SR // VID_FRAME_RATE
assert((22050/VID_FRAME_RATE) % 1 == 0)

def get_center(region):
    return int(region['x'] + 0.5*region['w']), int(region['y'] + 0.5*region['h'])


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

def process_rows(model, dataFrame):
    batch_size = 32
    output_dir = 'vids_2'
    DIST_THRESH = 20
    DIST_COEF = 3
    for row in dataFrame:
        # Download video and make directories
        try:

            vid_id = f"{row[1]}_{str(row[2])}"
            vid_dir = os.path.join(output_dir, vid_id)
            if not os.path.exists(vid_dir):
                os.mkdir(vid_dir)
            vid_path = os.path.join(output_dir, vid_id, f'{vid_id}.mp4')
            wav_path = os.path.join(output_dir, vid_id, f'{vid_id}.wav')
            if not os.path.exists(vid_path):
                command = f'yt-dlp --extractor-args "youtube:player_client=web" -f "(bestvideo+bestaudio/best)[protocolup=dash]" -q -o {vid_id}.mp4 --download-sections *{row[2]}-{row[3]} --format 18 -P {vid_dir}  "https://www.youtube.com/watch?v={vid_id}"'
                subprocess.run(shlex.split(command))
            if not os.path.exists(vid_path):
                os.rmdir(vid_dir)
            else:
                # Extract audio and facial features (nill for now)
                if not os.path.exists(wav_path):
                    subprocess.run(['ffmpeg', '-i', vid_path, '-ac', '1',  wav_path,'-loglevel', 'warning'])
                
                
                # Extract frames from video
                frames = []
                video_stream = cv2.VideoCapture(vid_path)
                found_correct_box = False
                while True:
                    still_reading, frame = video_stream.read()
                    
                    
                    if not still_reading:
                        video_stream.release()
                        break
                    frames.append(frame)
                    
                    [x, y, _] = frame.shape
                    head_center = np.array([row[4]*y, row[5]*x])
                    if not found_correct_box:
                        objs = DeepFace.analyze(img_path = frame, 
                            detector_backend='yolov8',
                            actions = ['age', 'gender', 'race', 'emotion'],
                            silent=True)
                        centers = [np.array(get_center(obj['region'])) for obj in objs]
                        dists = [np.linalg.norm(center - head_center) for center in centers]
                        if (min(dists) < DIST_THRESH): 
                            if (len(dists) == 1) or (dists[np.argsort(dists)[1]] > DIST_COEF*min(dists)):
                                correct_box = objs[np.argmin(dists)]
                                found_correct_box = True
                video_stream.release()
                cv2.destroyAllWindows()
                facial_attributes = {'race': correct_box['race'], 'gender': correct_box['gender'], 'emotion': correct_box['emotion'], 'age': correct_box['age'], 'lang': row[6]}
                json.dump(facial_attributes, open(f'{vid_dir}/{vid_id}_feat.json', 'w'))

                # Make ground truth data, only take as many frames as we have
                ground_truth = extract_features(wav_path)[:, :len(frames)]
                np.save(f'{vid_dir}/{vid_id}_pros.npy', ground_truth)

                # Run YOLO on all frames
                
                batches = [frames[i:i + batch_size] for i in range(0, len(frames), batch_size)]
                counter, failed_detection = 0, False
                frames = []
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
                            if (min(dists) < DIST_THRESH) or (dists[np.argsort(dists)[1]] > DIST_COEF*min(dists)):
                                correct_box = boxes[np.argmin(dists)]
                            else:
                                failed_detection = True
                        x1, y1, x2, y2 = [round(x) for x in correct_box]
                        frames.append(cv2.resize(result.orig_img[y1:y2,x1:x2, :], (96, 96)))
                        #cv2.imwrite(f'{vid_dir}/{vid_id}_{counter}.jpg', )
                        counter += 1
                np.save(f'{vid_dir}/{vid_id}_frames.npy', np.array(frames))
                if failed_detection:
                    print("Could not confidently detect faces in all frames for", vid_dir)
                    shutil.rmtree(vid_dir)
        except Exception as e:
            print(e)
            shutil.rmtree(vid_dir)
            print("Something failed for row:", row[0])
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Preprocess 10 rows')
    parser.add_argument('-s', '--start', help = 'starting point', required=True)
    parser.add_argument('-c', '--count', help = 'how many rows to process', required=True)
    args = parser.parse_args()
    start = int(args.start)
    count = int(args.count)
    start_time = time.time()
    raw_data = pd.read_csv('data/avspeech_test_langs.csv')
    dataFrame = np.array(raw_data)[start:start+count]
    model = YOLO('yolov8m-face.pt')
    process_rows(model, dataFrame)
    print(f'Processing time for rows {start} to {start + count - 1} is {time.time() - start_time} seconds.\n')
