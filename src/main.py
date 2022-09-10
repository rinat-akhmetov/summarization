from fire import Fire
from text.export_processors import TextExportProcessor
from text.summarizers.openai_adapter import OpenAIProcessor, PromtProcessor
from text.zoom.processor import ZoomTranscriptionProcessor


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


if __name__ == '__main__':
    Fire({
        'process_transcript': process_transcript,
    })
