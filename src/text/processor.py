class BaseProcessor:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class BaseTranscriptionProcessor(BaseProcessor):
    def __call__(self, transcription_path: str) -> list[str]:
        try:
            with open(transcription_path, 'r') as f:
                transcriptions = f.readlines()
        except FileNotFoundError:
            raise FileNotFoundError(f"File {transcription_path} not found")
        return transcriptions
