"""
Veri seti şeması ve etiket sabitleri.
Makaledeki deney tasarımına uygun sample ve split yapısı.
"""
from __future__ import annotations

from typing import Any

# Ana etiket sınıfları (label_main)
LABEL_REAL_SYNC = "real_sync"
LABEL_FAKE_SYNC_SHIFT = "fake_sync_shift"
LABEL_FAKE_CONTENT_MISMATCH = "fake_content_mismatch"
LABEL_FAKE_AUDIO_SYNTHETIC = "fake_audio_synthetic"
LABEL_FAKE_VISUAL = "fake_visual"

LABEL_MAIN_CHOICES = [
    LABEL_REAL_SYNC,
    LABEL_FAKE_SYNC_SHIFT,
    LABEL_FAKE_CONTENT_MISMATCH,
    LABEL_FAKE_AUDIO_SYNTHETIC,
    LABEL_FAKE_VISUAL,
]

# Senkron kaydırma değerleri (ms)
SYNC_SHIFT_MS_OPTIONS = [100, 200, 300, 400, 500]

# Uyumsuzluk türleri (mismatch_type)
MISMATCH_TEMPORAL = "temporal_shift"
MISMATCH_CONTENT = "content_mismatch"
MISMATCH_SYNTHETIC_AUDIO = "synthetic_audio"
MISMATCH_VISUAL = "visual_manipulation"
MISMATCH_NONE = "none"


def sample_schema(
    sample_id: str,
    speaker_id: str,
    video_path: str,
    audio_path: str,
    transcript_tr: str,
    label_main: str,
    label_visual_fake: int = 0,
    label_audio_fake: int = 0,
    label_sync: int = 1,
    sync_shift_ms: int | None = None,
    mismatch_type: str = MISMATCH_NONE,
    domain: str = "academic",
    duration_sec: float = 0.0,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Tek bir örnek için standart metadata sözlüğü.
    """
    return {
        "sample_id": sample_id,
        "speaker_id": speaker_id,
        "video_path": video_path,
        "audio_path": audio_path,
        "transcript_tr": transcript_tr,
        "label_main": label_main,
        "label_visual_fake": int(label_visual_fake),
        "label_audio_fake": int(label_audio_fake),
        "label_sync": int(label_sync),
        "sync_shift_ms": sync_shift_ms,
        "mismatch_type": mismatch_type,
        "domain": domain,
        "duration_sec": float(duration_sec),
        **kwargs,
    }


def is_real_sync(s: dict[str, Any]) -> bool:
    return s.get("label_main") == LABEL_REAL_SYNC and s.get("label_sync") == 1


def is_visual_fake(s: dict[str, Any]) -> bool:
    return s.get("label_visual_fake", 0) == 1


def is_audio_fake(s: dict[str, Any]) -> bool:
    return s.get("label_audio_fake", 0) == 1


def is_sync_ok(s: dict[str, Any]) -> bool:
    return s.get("label_sync", 0) == 1
