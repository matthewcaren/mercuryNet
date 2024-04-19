import pandas as pd
import numpy as np
import subprocess
import shlex
import time
import os
import json
from moviepy.editor import VideoFileClip
import cv2
from ultralytics import YOLO
import librosa

model = YOLO('yolov8m-face.pt')
def extract_audio_from_mp4(input_file, output_file):
    video = VideoFileClip(input_file)
    audio = video.audio
    audio.write_audiofile(output_file)
    video.close()

# constants
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

def resynthesize(features, max_length=None):
    '''
    resynthesize audio from features
    expects features = (features, time) array, with features being stacked (f0, voiced, rms)
    '''
    TOTAL_LEN = features.shape[-1]*HOP_SIZE if max_length==None else min(features.shape[-1]*HOP_SIZE, max_length)
    omega = np.zeros_like(TOTAL_LEN)
    for i in range(features.shape[-1] - 1):
        last_omega = omega[i*HOP_SIZE - 1]
        omega[i*HOP_SIZE : (i+1)*HOP_SIZE] = np.arange(HOP_SIZE)/AUDIO_SR*2*np.pi * features[0,i] + last_omega
    
    amplitudes = np.interp(x = np.linspace(0, 1, num=features.shape[-1]*HOP_SIZE),
                           xp = np.linspace(0, 1, num=features.shape[-1]),
                           fp = features[2,:])
    
    return np.sin(omega)[:TOTAL_LEN]**5 * amplitudes[:TOTAL_LEN]


def process_rows(dataFrame):
    batch_size = 32
    output_dir = './vids'
    for row in dataFrame:
        vid_id = row[0]
        vid_dir = os.path.join(output_dir, vid_id)
        if not os.path.exists(vid_dir):
            os.mkdir(vid_dir)
        vid_path = os.path.join(output_dir, vid_id, f'{vid_id}.mp4')
        wav_path = os.path.join(output_dir, vid_id, f'{vid_id}.wav')
        if not os.path.exists(vid_path):
            command = f'yt-dlp -q -f "(bestvideo+bestaudio/best)[protocolup=dash]" -o {vid_id}.mp4 --download-sections *{row[1]}-{row[2]} --format 18 -P {vid_dir}  "https://www.youtube.com/watch?v={vid_id}"'
            subprocess.run(shlex.split(command))
        if not os.path.exists(wav_path):
            extract_audio_from_mp4(vid_path, wav_path)
        facial_attributes = {'race': 0, 'gender': 0, 'emotion': 0, 'age': 0}
        json.dump(facial_attributes, open(f'{vid_dir}/{vid_id}_feat.json', 'w'))

        frames = []
        video_stream = cv2.VideoCapture(vid_path)
        while True:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            shape = frame.shape
            x, y = shape[0], shape[1]
            frames.append(frame)

        num_frames = len(frames)
        ground_truth = extract_features(wav_path)
        ground_truth = ground_truth[:, :num_frames]
        np.save(f'{vid_dir}/{vid_id}_pros.npy', ground_truth)

        head_center = np.array([row[3]*y, row[4]*x])
        batches = [frames[i:i + batch_size] for i in range(0, len(frames), batch_size)]
        counter = 0
        for batch in batches:
            results = model.predict(batch, device='mps', verbose=False)
            for result in results:
                orig_img = result.orig_img
                boxes = result.boxes.xyxy.cpu().numpy()
                if len(boxes) == 0:
                    raise Exception("No faces detected")
                elif len(boxes) == 0:
                    correct_box = boxes[0]
                else:
                    centers = [np.array([0.5*(box[2] + box[0]), 0.5*(box[3] + box[1])]) for box in boxes]
                    dists = [np.linalg.norm(center - head_center) for center in centers]
                    if min(dists) > 20:
                        raise Exception("Warning, multiple faces detected but no close face")
                    correct_box = boxes[np.argmin(dists)]
                x1, y1, x2, y2 = [round(x) for x in correct_box]
                cv2.imwrite(f'{vid_dir}/{vid_id}_{counter}.jpg', orig_img[y1:y2,x1:x2, :])
                counter += 1

start = time.time()
data = pd.read_csv('./data/avspeech_test.csv')
data = np.array(data)[9:10]
process_rows(data)
print(time.time() - start)