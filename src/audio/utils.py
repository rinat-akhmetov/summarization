from glob import glob
from pathlib import Path


def get_list_of_audio(zoom_path: Path):
    audio_recording = zoom_path / 'Audio Record'
    # get all the audio files
    audio_files = glob(str(audio_recording / '*.m4a'))
    print('Found {} audio files'.format(len(audio_files)))
    return audio_files

