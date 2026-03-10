"""
Türkçe konuşma transkripsiyonu. Whisper veya yerel ASR kullanılabilir.
"""
from pathlib import Path


def transcribe_whisper(audio_path, model_size: str = "base", language: str = "tr") -> str:
    """
    OpenAI Whisper ile Türkçe transkript.
    Kurulum: pip install openai-whisper
    """
    try:
        import whisper
    except ImportError:
        return ""
    model = whisper.load_model(model_size)
    result = model.transcribe(str(audio_path), language=language)
    return (result.get("text") or "").strip()


def transcribe_tr(audio_path, use_whisper: bool = True) -> str:
    """
    Türkçe ses dosyasından metin üretir.
    use_whisper=True ise Whisper kullanır; yoksa boş string (stub).
    """
    if use_whisper:
        return transcribe_whisper(audio_path)
    return ""


def transcribe_tr_with_timestamps(audio_path):
    """
    Opsiyonel: segment bazlı timestamp'li transkript.
    Whisper segment kullanılabilir.
    """
    try:
        import whisper
    except ImportError:
        return []
    model = whisper.load_model("base")
    result = model.transcribe(str(audio_path), language="tr")
    segments = result.get("segments") or []
    return [{"start": s["start"], "end": s["end"], "text": s.get("text", "").strip()} for s in segments]
