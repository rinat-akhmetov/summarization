# SPDX-License-Identifier: Apache-2.0

"""
Purpose

Shows how to use AWS SDK for Python (Boto3) to call Amazon Transcribe to make a
transcription of an audio file.

This script is intended to be used with the instructions for getting started in the
Amazon Transcribe Developer Guide here:
    https://docs.aws.amazon.com/transcribe/latest/dg/getting-started.html.
"""
import os
from pathlib import Path

from fire import Fire
from google.cloud import speech
from google.cloud import storage

# from audio.transcriber import Transcriber

os.environ['AWS_PROFILE'] = 'EDU'
os.environ['AWS_REGION'] = 'us-east-1'
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'

class Transcriber:
    def __init__(self, file_path: str):
        """
        :param file_path: Path to the file to transcribe s3 or gcp.
        """
        assert file_path is not None
        assert file_path.startswith('s3://') or file_path.startswith('gs://')
        self.file_path: str = file_path

    def transcribe(self):
        raise NotImplementedError

def upload_blob(source_file_name):
    """Uploads a file to the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The path to your file to upload
    # source_file_name = "local/path/to/file"
    # The ID of your GCS object
    # destination_blob_name = "storage-object-name"
    bucket_name = 'home-experiments'

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    destination_blob_name = Path(source_file_name).name
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )

def upload(file_path: Path):
    storage_client = storage.Client()

    buckets = list(storage_client.list_buckets())

    bucket = storage_client.get_bucket("home-experiments")

    blob = bucket.blob(f'{file_path.name}')
    blob.upload_from_filename(file_path)
    print(buckets)


class GoogleTranscriber(Transcriber):
    def __init__(self, file_path: str):
        """
        :param file_path: Path to the file to transcribe s3 or gcp.
        """
        super().__init__(file_path)
        assert self.file_path.startswith('gs://'), 'File path must start with gs://'
        self.project_name = Path(self.file_path).name
        self.language = 'en-US'
        self.language = 'ru-RU'
        self.job_name = f'{self.project_name}-{self.language}'
        self.client = speech.SpeechClient()
        self.artifacts_path = 'artifacts/transcriptions/google'
        if not os.path.exists(self.artifacts_path):
            os.makedirs(self.artifacts_path)

    def transcribe(self):
        """Transcribe the given audio file asynchronously."""

        """Asynchronously transcribes the audio file specified by the gcs_uri."""

        client = speech.SpeechClient()

        audio = speech.RecognitionAudio(uri=self.file_path)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.FLAC,
            sample_rate_hertz=32000,
            language_code=self.language,
        )

        operation = client.long_running_recognize(config=config, audio=audio)

        print("Waiting for operation to complete...")
        timeout = 60 * 60
        response = operation.result(timeout=timeout)

        # Each result is for a consecutive portion of the audio. Iterate through
        # them to get the transcripts for the entire audio file.
        for result in response.results:
            # The first alternative is the most likely one for this portion.
            print(u"Transcript: {}".format(result.alternatives[0].transcript))
            print(u"Confidence: {}".format(result.alternatives[0].confidence))


if __name__ == '__main__':
    file_path = 'gs://home-experiments/audioRinatAkhmetov31997267099.m4a'
    transcriber = GoogleTranscriber(file_path)
    transcriber.transcribe()