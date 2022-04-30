import hashlib
import pickle
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


def cache(function):
    def wrapper(promt):
        hash_object = hashlib.sha256(bytes(promt, "utf-8"))
        h = hash_object.hexdigest()
        Path('cache').mkdir(exist_ok=True)
        if not glob(f'cache/{function.__name__}_{h}.pickle'):
            print('Cache miss. Making new request.')
            response = function(promt)
            print('Caching...')
            print(f'to cache/{function.__name__}_{h}.pickle')
            with open(f'cache/{function.__name__}_{h}.pickle', 'wb') as f:
                pickle.dump(response, f)
        else:
            print('Cache hit.')
            print(f'from cache/{function.__name__}_{h}.pickle')
            with open(f'cache/{function.__name__}_{h}.pickle', 'rb') as f:
                response = pickle.load(f)
        return response

    return wrapper
