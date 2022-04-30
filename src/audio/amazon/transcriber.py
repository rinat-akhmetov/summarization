# SPDX-License-Identifier: Apache-2.0

"""
Purpose

Shows how to use AWS SDK for Python (Boto3) to call Amazon Transcribe to make a
transcription of an audio file.

This script is intended to be used with the instructions for getting started in the
Amazon Transcribe Developer Guide here:
    https://docs.aws.amazon.com/transcribe/latest/dg/getting-started.html.
"""
import json
import os
import time
from pathlib import Path
from typing import Optional

import boto3
import requests
from requests import Response
from tqdm import tqdm

from audio.transcriber import Transcriber
import logging

os.environ['AWS_PROFILE'] = 'EDU'
os.environ['AWS_REGION'] = 'us-east-1'
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'


class AWSTranscriber(Transcriber):
    def __init__(self, file_path: str):
        """
        :param file_path: Path to the file to transcribe s3 or gcp.
        """
        super().__init__(file_path)
        assert self.file_path.startswith('s3://'), 'File URI must start with s3://'
        self.project_name = Path(self.file_path).name
        self.language = 'en-US'
        self.language = 'ru-RU'
        self.job_name = f'{self.project_name}-{self.language}'
        self.artifacts_path = 'artifacts/transcriptions/amazon'
        self.transcribe_client = boto3.client('transcribe')

        if not os.path.exists(self.artifacts_path):
            os.makedirs(self.artifacts_path)

    def get_job(self) -> Optional[dict]:
        try:
            job = self.transcribe_client.get_transcription_job(TranscriptionJobName=self.job_name)
        except Exception as e:
            logging.info(f"Job {self.job_name} does not exist.")
            return None
        return job

    def transcribe_request(self):
        job = self.transcribe_client.start_transcription_job(
            TranscriptionJobName=self.job_name,
            Media={'MediaFileUri': self.file_path},
            MediaFormat='mp4',
            LanguageCode=self.language,
            Settings={
                'ShowSpeakerLabels': True,
                'MaxSpeakerLabels': 10,
                'ShowAlternatives': False,
                # 'MaxAlternatives': 123,
            },
        )
        return job

    def get_job_result(self) -> tuple[Optional[Response], str]:
        job = self.get_job()
        job_status = job['TranscriptionJob']['TranscriptionJobStatus']
        if job_status in ['COMPLETED', 'FAILED']:
            logging.info(f"Job {self.job_name} is {job_status}.")
            if job_status == 'COMPLETED':
                logging.info(f"Job {self.job_name} completed.")
                logging.info(
                    f"Download the transcript from\n"
                    f"\t{job['TranscriptionJob']['Transcript']['TranscriptFileUri']}.")
                response = requests.get(job['TranscriptionJob']['Transcript']['TranscriptFileUri'])
                return response, job_status
            else:
                logging.error('Job failed.')
                logging.error(f"Error message: {job['TranscriptionJob']['FailureReason']}")
            return None, job_status
        return None, job_status

    def export(self, response):
        logging.info("Saving transcript to file.")
        logging.info(f"{self.artifacts_path}/{self.job_name}.json")
        with open(f'{self.artifacts_path}/{self.job_name}.json', 'w') as f:
            json.dump(response.json(), f)

    def transcribe_file(self):
        job = self.get_job()
        if job is None:
            self.transcribe_request()
        max_tries = 60
        pbar = tqdm(total=max_tries)
        while max_tries > 0:
            max_tries -= 1
            job_result, job_status = self.get_job_result()
            if job_result:
                self.export(job_result)
            pbar.set_description(f'{self.job_name} status: {job_status}, tries: {max_tries}')
            time.sleep(10)

    def transcribe(self):
        logging.info('Transcribing file:', self.file_path)
        return self.transcribe_file()
