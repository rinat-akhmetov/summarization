class Transcriber:
    def __init__(self, file_path: str):
        """
        :param file_path: Path to the file to transcribe s3 or gcp.
        """
        assert file_path is not None
        assert file_path.startswith('s3://') or file_path.startswith('gs://')
        self.file_path: str = file_path

    def transcribe(self):
        raise NotImplementedError
