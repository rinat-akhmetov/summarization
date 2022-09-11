import os
from pathlib import Path

import streamlit as st

from text.summarizers.openai_adapter import PromtProcessor, OpenAIProcessor
from text.zoom.processor import ZoomTranscriptionProcessor


def process_transcript(transcript_path):
    best_of = 1
    max_tokens = 400
    limit = 4001
    promt_processor = PromtProcessor(best_of, max_tokens, limit)
    openai_processor = OpenAIProcessor()
    pipeline = [
        ZoomTranscriptionProcessor(),
        promt_processor, openai_processor,
        promt_processor,
        openai_processor,
    ]
    args = transcript_path
    for step in pipeline:
        args = step(args)
    return args


def file_uploader_component(parent_component):
    uploaded_file = parent_component.file_uploader("Transcription", type=['vtt'])
    video_path = None
    if uploaded_file is not None:
        is_valid = True
        with st.spinner(text='Uploading...'):
            Path(os.path.join("data", "transcriptions")).mkdir(parents=True, exist_ok=True)

            with open(os.path.join("data", "transcriptions", uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
            video_path = f'data/transcriptions/{uploaded_file.name}'
    else:
        is_valid = False
    return is_valid, video_path


def main():
    st.title("TL;DR")
    st.sidebar.title("Features")
    info = st.container()
    is_valid, transcription_path = file_uploader_component(st.sidebar)
    if is_valid:
        st.session_state.transcription_path = transcription_path
        with info:
            with st.spinner(text='Process Transcription'):
                promts = process_transcript(st.session_state.transcription_path)
                st.session_state.promts = promts
            if 'promts' in st.session_state:
                source_texts = []
                for i, promt in enumerate(st.session_state.promts):
                    source_texts.append(
                        st.markdown(promt)
                    )
                st.download_button('Download tldr', '\n'.join(promts), file_name='tldr.txt')

    pass


if __name__ == '__main__':
    main()
