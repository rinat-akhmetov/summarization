import json
from glob import glob
from pathlib import Path

from fire import Fire
from joblib import Parallel, delayed

from audio.amazon.subtitles import create_subtitle, create_subtitles_file
from audio.amazon.transcriber import Transcriber
from audio.utils import get_list_of_audio
from text.main import prepare_tldr, generate_promts_from_transcription, merge_responses, beautify_lines, \
    key_points_extraction
from text.openai import request_open_ai_meeting_notes, key_points
from utils.utils import upload_file_to_s3


def process_transcript(transcript_path: str = "./artifacts/GMT20220414-171026_Recording.transcript.vtt") -> None:
    """
    Processes a transcript.
    """
    promts = generate_promts_from_transcription(transcript_path)

    responses = request_open_ai_meeting_notes(promts)
    result_string = merge_responses(responses)
    beautified_result_string = beautify_lines(result_string)
    with open(transcript_path + '_result.txt', 'w') as result_file:
        result_file.write(beautified_result_string)
    prepare_tldr(beautified_result_string)


def generate_transcriptions(transcription_folder_path: Path = None):
    transcription_paths = glob(str(Path(transcription_folder_path) / '*.json'))
    grouped_labels = Parallel(n_jobs=8)(
        delayed(generate_transcription)(transcription_path) for transcription_path in transcription_paths)
    flatten_labels = [label for labels in grouped_labels for label in labels]
    flatten_labels = sorted(flatten_labels, key=lambda x: x.start_time)
    subtitles_file_path = str(transcription_folder_path).replace('transcriptions', 'subtitles')
    subtitles_file_path = Path(subtitles_file_path) / 'transcription.srt'
    create_subtitles_file(subtitles_file_path, flatten_labels)


def generate_transcription(transcription_path: Path = None):
    with open(transcription_path, 'r') as f:
        response = json.load(f)
    speaker_name = Path(transcription_path).stem.split('audio')[1].split('m4a')[0][:-12]
    grouped_items = create_subtitle(response, speaker_name)
    return grouped_items


def transcribe_wrapper(audio_path: str):
    return Transcriber(audio_path).transcribe()


def create_transcription(audio_paths: list[str]):
    print('Processing audio files')
    Parallel(n_jobs=2)(delayed(transcribe_wrapper)(audio_path) for audio_path in audio_paths)


def process_audios(zoom_path: Path = Path(
    '/Users/arrtz3/Documents/Zoom/2022-04-29 12.04.35 Google Calendar Meeting (not synced)')):
    audio_paths = get_list_of_audio(zoom_path)
    s3_paths = Parallel(n_jobs=5)(delayed(upload_file_to_s3)(audio_path) for audio_path in audio_paths)
    create_transcription(s3_paths)


def write_key_points(conversation_path):
    conversation_path = Path(conversation_path)
    for file_name in conversation_path.glob('*.txt'):
        with open(file_name, 'r') as f:
            input_text = f.read()
        response = key_points(input_text)
        file_name = str(file_name)
        with open(file_name.replace('.txt', '_key_points.txt'), 'w') as f:
            open_ai_text = key_points_extraction(response)
            f.write(open_ai_text)


if __name__ == '__main__':
    Fire({
        'process_audios': process_audios,
        'process_transcript': process_transcript,
        'key_points': write_key_points,
        'generate_transcriptions': generate_transcriptions
    })
