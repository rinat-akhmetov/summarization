import logging
import os

import openai
from joblib import delayed, Parallel
from nltk import sent_tokenize
from openai.openai_object import OpenAIObject

from text.processor import BaseProcessor
from utils.utils import cache, Config

openai.api_key = os.getenv("OPENAI_API_KEY")


class OpenAIProcessor(BaseProcessor):
    def __init__(self, config: Config = None):
        if config is None:
            config = Config(
                engine="text-davinci-002",
                temperature=0.0,
                frequency_penalty=1,
                presence_penalty=1,
                key_phrase="Describe the key points of the discussion from the transcription of the meeting:\n",
                best_of=1,
                max_tokens=400
            )
        self.config = config

    def __call__(self, promts: list[str]) -> list[str]:
        responses = request_open_ai(promts, self.config)
        texts = [response['choices'][0]['text'] for response in responses]
        return texts


class PromtProcessor(BaseProcessor):
    def __init__(self, best_of, max_tokens, limit):
        self.best_of = best_of
        self.max_tokens = max_tokens
        self.limit = limit

    def __call__(self, promts: list[str]) -> list[str]:
        promt = '\n'.join(promts)
        sentences = sent_tokenize(promt)
        promts = [sentences[0]]
        for line in sentences[1:]:
            if len(promts[-1]) + len(line) + self.best_of * self.max_tokens < self.limit:
                promts[-1] += line
            else:
                promts.append(line)
        return promts


def request_open_ai(promts, config: Config, n_jobs=10) -> list:
    logging.info('Requesting OpenAI notes')
    responses = Parallel(n_jobs=n_jobs)(delayed(openai_request)(promt, config) for promt in promts)
    return responses


@cache
def openai_request(promt: str, config: Config) -> None:
    promt = config.key_phrase + promt
    response = openai.Completion.create(
        engine=config.engine,
        prompt=promt,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        best_of=config.best_of,
        top_p=1,
        frequency_penalty=config.frequency_penalty,
        presence_penalty=config.presence_penalty
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
