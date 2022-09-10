import hashlib
import json
import logging
import pickle
from collections import namedtuple
from glob import glob
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

def upload_file_to_s3(audio_path: str):
    bucket_name = 'lokoai-lambdas-demo'
    s3_client = boto3.client('s3')
    project_name = Path(audio_path).name
    if not check_s3_file(bucket_name, project_name):
        print('uploading {} to s3'.format(project_name))
        s3_client.upload_file(audio_path, bucket_name, project_name)
    return f's3://{bucket_name}/{project_name}'


def check_s3_file(bucket_name, project_name):
    s3_client = boto3.client('s3')
    try:
        return s3_client.head_object(Bucket=bucket_name, Key=project_name)
    except ClientError:
        # Not found
        pass
    return False


# temperature number Optional Defaults to 1
# What sampling temperature to use. Higher values means the model will take more risks.
# Try 0.9 for more creative applications, and 0 (argmax sampling) for ones with a well-defined answer.
# We generally recommend altering this or top_p but not both.

# presence_penalty number Optional Defaults to 0
# Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far,
# increasing the model's likelihood to talk about new topics.


# frequency_penalty number Optional Defaults to 0
# Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far,
# decreasing the model's likelihood to repeat the same line verbatim.
Config = namedtuple(
    'Config',
    [
        'engine',
        'temperature',
        'key_phrase',
        'frequency_penalty',
        'presence_penalty',
        'max_tokens',
        'best_of',
    ]
)


def process_arguments(args: list) -> list[str]:
    results = []
    for arg in args:
        if type(arg) == str:
            results.append(arg)
        elif type(arg) == Config:
            results.append(json.dumps(arg))
    return results


def cache(function):
    def wrapper(*args, **kwargs):
        _args = process_arguments(args)
        hash_object = hashlib.sha256(bytes(''.join(_args), "utf-8"))
        h = hash_object.hexdigest()
        Path("cache").mkdir(parents=True, exist_ok=True)
        if not glob(f'cache/{function.__name__}_{h}.pickle'):
            print('Cache miss. Making new request.')
            response = function(*args, **kwargs)
            logging.debug('Caching...')
            logging.debug(f'to cache/{function.__name__}_{h}.pickle')
            with open(f'cache/{function.__name__}_{h}.pickle', 'wb') as f:
                pickle.dump(response, f)
        else:
            print(f'from cache/{function.__name__}_{h}.pickle')
            with open(f'cache/{function.__name__}_{h}.pickle', 'rb') as f:
                response = pickle.load(f)
        return response

    return wrapper
