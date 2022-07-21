import os

import openai
from joblib import delayed, Parallel
from openai.openai_object import OpenAIObject
from tqdm import tqdm

from utils.utils import cache

openai.api_key = os.getenv("OPENAI_API_KEY")
MAX_TOKENS = 400
BEST_OF = 2
LIMIT = 4001


def request_open_ai_meeting_notes(promts) -> list:
    responses = Parallel(n_jobs=5)(delayed(open_ai_meeting_notes)(promt) for promt in tqdm(promts))
    return responses


@cache
def open_ai_meeting_notes(promt: str, temperature=0.1, frequency_penalty=0.01, presence_penalty=0.1) -> None:
    promt = "Convert workshop meeting into a first-hand account of the meeting:\n\n" + promt
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=promt,
        temperature=temperature,
        max_tokens=MAX_TOKENS,
        best_of=BEST_OF,
        top_p=1,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty
    )
    return response


@cache
def openai_tldr(promt: str) -> OpenAIObject:
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=promt + "\n\nTl;dr",
        temperature=0.7,
        max_tokens=MAX_TOKENS,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response


@cache
def key_points(promt: str, temperature=0.7, frequency_penalty=0, presence_penalty=0) -> OpenAIObject:
    promt = "It is the part of the conversation zoom conversation:\n\n" + promt
    promt += '\n\nWhat are the key points of the conversation?'
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=promt,
        temperature=temperature,
        max_tokens=MAX_TOKENS,
        best_of=BEST_OF,
        top_p=1,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty
    )
    return response
