from random import choice

import nltk
from joblib import Parallel, delayed
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

from text.openai import BEST_OF, MAX_TOKENS, openai_tldr, LIMIT

nltk.download('punkt')

key_phrase = 'At the meeting,'
possible_replacement = ['And also,', "Besides,", "In Addition", "Furthermore"]


def prepare_text_for_request(promt: str):
    sentences = sent_tokenize(promt)
    promts = [sentences[0]]
    for line in sentences[1:]:
        if len(promts[-1]) + len(line) + BEST_OF * MAX_TOKENS < LIMIT:
            promts[-1] += line
        else:
            promts.append(line)
    return promts


def prepare_tldr(promt: str):
    promts = prepare_text_for_request(promt)
    text = ''
    responses = Parallel(n_jobs=5)(delayed(openai_tldr)(promt) for promt in tqdm(promts))
    for response in responses:
        text += response.choices[0].text + '\n'
    beautified_result_string = beautify_lines(text)
    export(beautified_result_string)


def export(beautified_result_string: str) -> None:
    with open('./tldr_result.txt', 'w') as result_file:
        result_file.write(beautified_result_string)


def beautify_lines(result_string, chars_in_a_line: int = 300) -> str:
    result_string = result_string.replace('\n\n', '\n')
    lines = result_string.split('\n')
    beautified_lines = []
    for line in lines:
        if len(line) > chars_in_a_line:
            sentences = sent_tokenize(line)
            _lines = [sentences[0]]
            for sentence in sentences[1:]:
                if len(_lines[-1]) + len(sentence) < chars_in_a_line:
                    _lines[-1] += sentence
                else:
                    _lines.append(sentence)
            beautified_lines += _lines
        else:
            beautified_lines.append(line)
    return '\n'.join(beautified_lines)


def merge_responses(responses) -> str:
    texts = []
    for i, response in enumerate(responses):
        text = response.choices[0].text
        if key_phrase in text:
            index = text.index(key_phrase)
            text = text[index:]
            texts.append(text)
        else:
            print(i, response.choices[0].finish_reason)
            print(response)
    if len(texts) == 0:
        return ''
    result_string = beautify_blocks(texts)
    return result_string


def beautify_blocks(texts):
    result_string = f'#{1}\n' + texts[0] + '\n'
    for i, text in tqdm(enumerate(texts[1:])):
        result_string += f'#{i + 2}\n'
        result_string += text.replace(key_phrase, choice(possible_replacement)).replace('\n\n', '\n') + '\n'
    return result_string


def generate_promts_from_transcription(transcript_path):
    with open(transcript_path, 'r') as transcript_file:
        transcript_lines = transcript_file.readlines()
        transcript_lines = transcript_lines[4::4]
        promts = [transcript_lines[0]]
        for line in transcript_lines[1:]:
            if len(promts[-1]) + len(line) + BEST_OF * MAX_TOKENS < 4001:
                promts[-1] += line
            else:
                promts.append(line)
    return promts
