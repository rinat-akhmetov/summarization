from text.processor import BaseTranscriptionProcessor


class ZoomTranscriptionProcessor(BaseTranscriptionProcessor):
    def __call__(self, transcription_path: str) -> list[str]:
        transcriptions = super().__call__(transcription_path)
        transcript_lines = transcriptions[4::4]
        return transcript_lines
