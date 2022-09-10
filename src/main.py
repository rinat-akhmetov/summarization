import json
from glob import glob
from pathlib import Path

from fire import Fire
from joblib import Parallel, delayed

from audio.amazon.subtitles import create_subtitle, create_subtitles_file
from audio.amazon.transcriber import Transcriber
from audio.utils import get_list_of_audio
from text.export_processors import TextExportProcessor
from text.summarizers.openai_adapter import OpenAIProcessor, PromtProcessor
from text.zoom.processor import ZoomTranscriptionProcessor
from utils.utils import upload_file_to_s3


def process_transcript(transcript_path: str = "./artifacts/GMT20220414-171026_Recording.transcript.vtt") -> None:
    """
    Processes a transcript.
    """
    best_of = 2
    max_tokens = 400
    limit = 4001
    promt_processor = PromtProcessor(best_of, max_tokens, limit)
    openai_processor = OpenAIProcessor()
    txt_processor = TextExportProcessor()
    pipeline = [
        ZoomTranscriptionProcessor(),
        promt_processor, openai_processor,
        promt_processor,
        openai_processor,
        txt_processor
    ]
    args = transcript_path
    for step in pipeline:
        args = step(args)
    print(args)


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


if __name__ == '__main__':
    Fire({
        'process_audios': process_audios,
        'process_transcript': process_transcript,
        'generate_transcriptions': generate_transcriptions
    })
