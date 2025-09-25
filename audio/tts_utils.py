from gtts import gTTS
from io import BytesIO

def text_to_speech_gtts(text: str):
    """Chuyển văn bản thành giọng nói bằng gTTS và trả về dưới dạng stream."""
    text = text.replace("*", "").replace("#", "")
    tts = gTTS(text=text, lang='vi', slow=False)
    audio_stream = BytesIO()
    tts.write_to_fp(audio_stream)
    audio_stream.seek(0)
    return audio_stream