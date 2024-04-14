from moviepy.editor import VideoFileClip
import os

def extract_audio_from_mp4(input_file, output_file):
    """
    Extracts the audio from an MP4 file and saves it as a WAV file.

    Args:
        input_file (str): The path to the input MP4 file.
        output_file (str): The path to the output WAV file.
    """
    video = VideoFileClip(input_file)
    audio = video.audio
    audio.write_audiofile(output_file)
    video.close()


if __name__ == "__main__":
    for i in range(17):

        input_file = f"./vids/{i}.mp4"
        output_file = f"./vids/{i}.wav"
        extract_audio_from_mp4(input_file, output_file)
