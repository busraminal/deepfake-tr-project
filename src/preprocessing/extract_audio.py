"""
Videodan ses çıkarır. data.yaml'daki audio_sr ve audio_format kullanılır.
"""
from pathlib import Path

import numpy as np
import soundfile as sf


def extract_audio_ffmpeg(video_path, out_path, sr=16000) -> bool:
    """FFmpeg ile videodan WAV çıkarır (tercih edilen)."""
    import subprocess
    video_path = Path(video_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(sr),
        str(out_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return out_path.exists()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def extract_audio_moviepy(video_path, out_path, sr=16000) -> bool:
    """MoviePy ile videodan ses çıkarır (ffmpeg yoksa)."""
    try:
        from moviepy.editor import VideoFileClip
    except ImportError:
        return False
    video_path = Path(video_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        clip = VideoFileClip(str(video_path))
        clip.audio.write_audiofile(str(out_path), fps=sr, verbose=False, logger=None)
        clip.close()
        return out_path.exists()
    except Exception:
        return False


def extract_audio(video_path, out_path, sr=16000) -> bool:
    """Videodan mono WAV çıkarır. Önce ffmpeg dene, yoksa moviepy."""
    if extract_audio_ffmpeg(video_path, out_path, sr):
        return True
    return extract_audio_moviepy(video_path, out_path, sr)


def load_audio(path, sr=None):
    """Ses dosyasını yükler (numpy, sample_rate)."""
    data, rate = sf.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr and rate != sr:
        import librosa
        data = librosa.resample(data.astype(np.float32), orig_sr=rate, target_sr=sr)
        rate = sr
    return data, rate
