import os
import tempfile
from pydub import AudioSegment

def convert_to_wav(input_path):
    """Chuyển đổi file audio sang định dạng WAV (16kHz, mono)"""
    try:
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        audio.export(temp_wav.name, format="wav")
        temp_wav.close()
        return temp_wav.name
    except Exception as e:
        print(f"Lỗi khi chuyển đổi file audio: {e}")
        return None

def split_audio_chunks(wav_path, chunk_length_ms):
    """Chia một file WAV thành các đoạn nhỏ."""
    audio = AudioSegment.from_wav(wav_path)
    chunks = []
    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i + chunk_length_ms]
        temp_chunk = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        chunk.export(temp_chunk.name, format="wav")
        temp_chunk.close()
        chunks.append(temp_chunk.name)
    return chunks

def transcribe_audio_chunks(model, chunk_paths):
    """Chuyển đổi các đoạn audio thành văn bản."""
    transcript = ""
    for chunk_path in chunk_paths:
        segments, _ = model.transcribe(chunk_path)
        for segment in segments:
            transcript += segment.text + " "
    return transcript.strip()