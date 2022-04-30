import json
from pathlib import Path
from typing import List

from fire import Fire
from tqdm import tqdm

from audio.amazon.transcriber import transcribe


class Alternative:
    confidence: float
    content: str

    def __init__(self, confidence: str, content: str):
        self.confidence = float(confidence)
        self.content = content


class Item:
    start_time: str
    end_time: str
    _type: str
    alternatives: List[Alternative]

    def content(self) -> str:
        return ' '.join([alternative.content for alternative in self.alternatives])

    def __init__(self, type: str, alternatives: List[Alternative], start_time: str = None, end_time: str = None):
        self.start_time = None
        if start_time is not None:
            self.start_time = float(start_time)
        self.end_time = None
        if end_time is not None:
            self.end_time = float(end_time)
        self._type = type
        self.alternatives = []
        self.speaker_label = None
        for alternative in alternatives:
            self.alternatives.append(Alternative(**alternative))


class SpeakerItem:
    start_time: float
    end_time: float
    speaker_label: str

    def __init__(self, start_time: str, end_time: str, speaker_label: str):
        self.start_time = float(start_time)
        self.end_time = float(end_time)
        self.speaker_label = speaker_label


class SpeakerLabel:
    start_time: float
    end_time: float
    speaker_label: str
    items: List[SpeakerItem]

    def __init__(self, start_time: str, end_time: str, speaker_label: str, items: List[SpeakerItem]):
        self.start_time = float(start_time)
        self.end_time = float(end_time)
        self.speaker_label = speaker_label
        self.items = []
        for item in items:
            self.items.append(SpeakerItem(**item))


def group_items_by_speaker(items: [Item]) -> [Item]:
    speakers: List[Item] = []
    current = items[0]
    result = [items[0]]
    for item in items[1:]:
        if item.start_time is None:
            result[-1].alternatives.extend(item.alternatives)
            # item.speaker_label = current.speaker_label
            continue
        if result[-1].speaker_label == item.speaker_label and item.start_time - result[-1].start_time < 5:
            result[-1].alternatives.extend(item.alternatives)
            result[-1].end_time = item.end_time
            continue
        result.append(item)

        # if current.speaker_label != item.speaker_label:
        #     speakers.append(current)
        #     current = item
        # else:
        #     current.end_time = item.end_time
        #     current.alternatives.extend(item.alternatives)
    return result


def format_time_for_subtitles(time: float) -> str:
    '00:00:01,840'
    'hh:mm:ss,ms'
    hours = int(time / 3600)
    minutes = int((time - hours * 3600) / 60)
    seconds = int(time - hours * 3600 - minutes * 60)
    milliseconds = int((time - hours * 3600 - minutes * 60 - seconds) * 1000)
    return f'{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}'


def create_subtitles_file(file_path: Path, grouped_items: [Item]):
    with open(file_path, 'w') as f:
        for index, item in enumerate(tqdm(grouped_items)):
            f.write(f'{index}\n')
            f.write(f'{format_time_for_subtitles(item.start_time)} --> {format_time_for_subtitles(item.end_time)}\n')
            f.write(f'{item.speaker_label}: {item.content()}\n\n')


def create_subtitle(response, speaker_name=None):
    items_dict = {}
    items = []
    for item in tqdm(response['results']['items']):
        try:
            i = Item(**item)
            items_dict[i.start_time] = i
            items.append(i)
        except Exception as e:
            print(e, item)

    speaker_labels = []
    for speaker_label in tqdm(response['results']['speaker_labels']['segments']):
        try:
            speaker_labels.append(SpeakerLabel(**speaker_label))
            if speaker_label:
                speaker_labels[-1].speaker_label = speaker_name
        except Exception as e:
            print(e, speaker_label)

    for speaker_label in speaker_labels:
        for item in speaker_label.items:
            items_dict[item.start_time].speaker_label = speaker_label.speaker_label
    grouped_items = group_items_by_speaker(items)
    return grouped_items


def main(file_path: str):
    # file_path = 'woman_speakers.json'
    # file_path = 's3://lokoai-lambdas-demo/AshleyMcArthur.mp4'
    if file_path.startswith('s3://'):
        response = transcribe(file_path)
    else:
        with open(file_path, 'r') as f:
            response = json.load(f)
    subtitles_file_path = Path(file_path).with_suffix('.srt')
    create_subtitle(response, subtitles_file_path)


if __name__ == '__main__':
    Fire(main)
