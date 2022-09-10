from text.processor import BaseProcessor


class TextExportProcessor(BaseProcessor):
    def __init__(self, file_name='meetings_notes.txt'):
        self.file_name = file_name

    def __call__(self, promts: list[str]) -> None:
        with open(self.file_name, 'w') as f:
            f.writelines('\n'.join(promts))
