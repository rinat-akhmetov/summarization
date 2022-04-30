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
from fire import Fire
from tqdm import tqdm

os.environ['AWS_PROFILE'] = 'EDU'
os.environ['AWS_REGION'] = 'us-east-1'
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
transcribe_client = boto3.client('transcribe')


def transcribe_file(job_name, file_uri, artifacts_path: Optional[str] = 'artifacts/transcriptions/amazon') -> Optional[
    dict]:
    language = 'en-US'
    language = 'ru-RU'
    job_name = f'{job_name}-{language}'
    job = check_the_job(job_name)
    if job is None:
        job = transcribe_client.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={'MediaFileUri': file_uri},
            MediaFormat='mp4',
            LanguageCode=language,
            Settings={
                'ShowSpeakerLabels': True,
                'MaxSpeakerLabels': 10,
                'ShowAlternatives': False,
                # 'MaxAlternatives': 123,
            },
        )
    max_tries = 60
    pbar = tqdm(total=max_tries)
    while max_tries > 0:
        max_tries -= 1
        job = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
        job_status = job['TranscriptionJob']['TranscriptionJobStatus']
        pbar.set_description(f'{job_name} status: {job_status}')
        pbar.update(1)
        if job_status in ['COMPLETED', 'FAILED']:
            print(f"Job {job_name} is {job_status}.")
            if job_status == 'COMPLETED':
                print(
                    f"Download the transcript from\n"
                    f"\t{job['TranscriptionJob']['Transcript']['TranscriptFileUri']}.")
                response = requests.get(job['TranscriptionJob']['Transcript']['TranscriptFileUri'])
                print("Saving transcript to file.")
                print(f"{artifacts_path}/{job_name}.json")
                with open(f'{artifacts_path}/{job_name}.json'.format(job_name=job_name), 'w') as f:
                    json.dump(response.json(), f)
                return response.content
            return None
        # else:
        # print(f"Waiting for {job_name}. Current status is {job_status}.")
        time.sleep(10)


def check_the_job(job_name: str) -> Optional[dict]:
    try:
        job = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
        job = job['TranscriptionJob']
        if job['TranscriptionJobStatus'] == 'FAILED':
            print(f"Job {job_name} failed.")
            print(f"Error message: {job['FailureReason']}")
        elif job['TranscriptionJobStatus'] == 'COMPLETED':
            print(f"Job {job_name} completed.")
        return job
    except Exception as e:
        print(f"Job {job_name} does not exist.")
    return None


def transcribe(file_uri: str):
    print('Transcribing file:', file_uri)
    project_name = Path(file_uri).name
    return transcribe_file(project_name, file_uri)


if __name__ == '__main__':
    Fire(transcribe)
