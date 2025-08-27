from gtts import gTTS
from io import BytesIO

def text_to_speech_gtts(text):
    """Chuyển văn bản thành giọng nói bằng gTTS và trả về dưới dạng stream."""
    tts = gTTS(text=text, lang='vi', slow=False)
    audio_stream = BytesIO()
    tts.write_to_fp(audio_stream)
    audio_stream.seek(0)
    return audio_stream