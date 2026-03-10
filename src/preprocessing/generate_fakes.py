"""
Kontrollü sahte örnek üretimi:
- fake_sync_shift: aynı sesi zamanda kaydır (100–500 ms)
- fake_content_mismatch: farklı cümle sesi bindir
- fake_audio_synthetic: TTS/sentetik ses (stub veya harici TTS)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from src.utils.schema import (
    sample_schema,
    LABEL_FAKE_SYNC_SHIFT,
    LABEL_FAKE_CONTENT_MISMATCH,
    LABEL_FAKE_AUDIO_SYNTHETIC,
    MISMATCH_TEMPORAL,
    MISMATCH_CONTENT,
    MISMATCH_SYNTHETIC_AUDIO,
    SYNC_SHIFT_MS_OPTIONS,
)
from src.utils.io import load_config, load_metadata, save_metadata, project_root


def shift_audio(audio_path, out_path, shift_ms: int, sr: int = 16000) -> bool:
    """Sesi zamanla kaydırır: shift_ms kadar boşluk ekleyip başa veya sonda keser."""
    audio_path = Path(audio_path)
    if not audio_path.is_file():
        # Preprocess başarısız olmuş veya audio_path boş; bu örnek için fake üretme.
        return False
    data, rate = sf.read(str(audio_path))
    if data.ndim > 1:
        data = data.mean(axis=1)
    if rate != sr:
        import librosa
        data = librosa.resample(data.astype(np.float32), orig_sr=rate, target_sr=sr)
        rate = sr
    n_shift = int(rate * shift_ms / 1000)
    # Sağa kaydır: başa sıfır ekle
    shifted = np.zeros(len(data) + n_shift, dtype=data.dtype)
    shifted[n_shift:] = data
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), shifted, rate)
    return True


def create_fake_sync_shift(
    real_meta: dict[str, Any],
    shift_ms: int,
    sample_id: str,
    config: dict[str, Any],
    base_dir: Path,
) -> dict[str, Any] | None:
    """Tek bir temporal shift sahte örneği üretir."""
    data_cfg = config.get("data", {})
    preprocess = config.get("preprocess", {})
    sr = preprocess.get("audio_sr", 16000)
    base_dir = Path(base_dir)
    audio_rel = real_meta.get("audio_path") or ""
    audio_in = base_dir / audio_rel if audio_rel else None
    fakes_dir = base_dir / data_cfg.get("fakes_dir", "data/processed/fakes")
    audio_out = fakes_dir / "audio" / f"{sample_id}.wav"
    if audio_in is None or not audio_in.exists():
        return None
    ok = shift_audio(audio_in, audio_out, shift_ms, sr)
    if not ok:
        return None
    out = sample_schema(
        sample_id=sample_id,
        speaker_id=real_meta["speaker_id"],
        video_path=real_meta["video_path"],
        audio_path=str(audio_out.relative_to(base_dir)),
        transcript_tr=real_meta["transcript_tr"],
        label_main=LABEL_FAKE_SYNC_SHIFT,
        label_visual_fake=0,
        label_audio_fake=0,
        label_sync=0,
        sync_shift_ms=shift_ms,
        mismatch_type=MISMATCH_TEMPORAL,
        domain=real_meta.get("domain", "academic"),
        duration_sec=real_meta.get("duration_sec", 0),
        source_sample_id=real_meta["sample_id"],
    )
    if "faces_dir" in real_meta:
        out["faces_dir"] = real_meta["faces_dir"]
    if "mouths_dir" in real_meta:
        out["mouths_dir"] = real_meta["mouths_dir"]
    return out


def create_fake_content_mismatch(
    real_meta_video: dict[str, Any],
    real_meta_audio: dict[str, Any],
    sample_id: str,
    config: dict[str, Any],
    base_dir: Path,
) -> dict[str, Any]:
    """
    Video A + Ses B. Ses dosyasını kopyala (videoyu değiştirmiyoruz, sadece metadata'da
    farklı transcript ve audio_path ile kayıt oluşturuyoruz).
    """
    base_dir = Path(base_dir)
    audio_rel = real_meta_audio.get("audio_path") or ""
    if not audio_rel:
        return None
    audio_src = base_dir / audio_rel
    fakes_dir = base_dir / config.get("data", {}).get("fakes_dir", "data/processed/fakes")
    audio_out = fakes_dir / "audio" / f"{sample_id}.wav"
    audio_out.parent.mkdir(parents=True, exist_ok=True)
    if not audio_src.exists():
        return None
    import shutil
    shutil.copy2(str(audio_src), str(audio_out))
    out = sample_schema(
        sample_id=sample_id,
        speaker_id=real_meta_video["speaker_id"],
        video_path=real_meta_video["video_path"],
        audio_path=str(audio_out.relative_to(base_dir)),
        transcript_tr=real_meta_audio["transcript_tr"],
        label_main=LABEL_FAKE_CONTENT_MISMATCH,
        label_visual_fake=0,
        label_audio_fake=0,
        label_sync=0,
        sync_shift_ms=None,
        mismatch_type=MISMATCH_CONTENT,
        domain=real_meta_video.get("domain", "academic"),
        duration_sec=real_meta_video.get("duration_sec", 0),
        source_video_id=real_meta_video["sample_id"],
        source_audio_id=real_meta_audio["sample_id"],
    )
    if "faces_dir" in real_meta_video:
        out["faces_dir"] = real_meta_video["faces_dir"]
    if "mouths_dir" in real_meta_video:
        out["mouths_dir"] = real_meta_video["mouths_dir"]
    return out


def create_fake_audio_synthetic_stub(
    real_meta: dict[str, Any],
    sample_id: str,
    config: dict[str, Any],
    base_dir: Path,
) -> dict[str, Any]:
    """
    Sentetik ses için stub: aynı sürede sessiz veya gürültü yazılabilir.
    Gerçek TTS entegrasyonu sonra eklenir.
    """
    base_dir = Path(base_dir)
    preprocess = config.get("preprocess", {})
    sr = preprocess.get("audio_sr", 16000)
    dur = real_meta.get("duration_sec", 5.0)
    fakes_dir = base_dir / config.get("data", {}).get("fakes_dir", "data/processed/fakes")
    audio_out = fakes_dir / "audio" / f"{sample_id}.wav"
    audio_out.parent.mkdir(parents=True, exist_ok=True)
    n = int(dur * sr)
    # Placeholder: sessiz (veya ileride TTS çıktısı yazılır)
    silent = np.zeros(n, dtype=np.float32)
    sf.write(audio_out, silent, sr)
    out = sample_schema(
        sample_id=sample_id,
        speaker_id=real_meta["speaker_id"],
        video_path=real_meta["video_path"],
        audio_path=str(audio_out.relative_to(base_dir)),
        transcript_tr="",
        label_main=LABEL_FAKE_AUDIO_SYNTHETIC,
        label_visual_fake=0,
        label_audio_fake=1,
        label_sync=0,
        sync_shift_ms=None,
        mismatch_type=MISMATCH_SYNTHETIC_AUDIO,
        domain=real_meta.get("domain", "academic"),
        duration_sec=dur,
        source_sample_id=real_meta["sample_id"],
    )
    if "faces_dir" in real_meta:
        out["faces_dir"] = real_meta["faces_dir"]
    if "mouths_dir" in real_meta:
        out["mouths_dir"] = real_meta["mouths_dir"]
    return out


def generate_all_fakes_for_real(
    real_meta: dict[str, Any],
    other_reals: list[dict[str, Any]],
    config: dict[str, Any],
    base_dir: Path,
) -> list[dict[str, Any]]:
    """
    Bir gerçek örnek için:
    - Her sync_shift_ms değerinde bir fake_sync_shift
    - Bir fake_content_mismatch (other_reals'tan rastgele ses)
    - Bir fake_audio_synthetic (stub)
    """
    base_dir = Path(base_dir)
    generated = []
    base_id = real_meta["sample_id"]

    for shift_ms in SYNC_SHIFT_MS_OPTIONS:
        sid = f"{base_id}_shift_{shift_ms}"
        m = create_fake_sync_shift(real_meta, shift_ms, sid, config, base_dir)
        if m:
            generated.append(m)

    if other_reals:
        import random
        other = random.choice(other_reals)
        if other["sample_id"] != base_id:
            sid = f"{base_id}_mismatch_{other['sample_id']}"
            m_mm = create_fake_content_mismatch(real_meta, other, sid, config, base_dir)
            if m_mm:
                generated.append(m_mm)

    sid_synth = f"{base_id}_synthetic"
    generated.append(
        create_fake_audio_synthetic_stub(real_meta, sid_synth, config, base_dir)
    )

    return generated


def run_generate_fakes(metadata_dir: str | Path, config_path: str | Path | None = None) -> list[dict]:
    """
    metadata_dir içindeki tüm real_sync *.json dosyalarını okuyup
    her biri için sahte örnekler üretir; metadata'ları fakes metadata klasörüne yazar.
    """
    base_dir = project_root()
    config = load_config(config_path or base_dir / "configs" / "data.yaml")
    metadata_dir = Path(metadata_dir)
    if not metadata_dir.is_absolute():
        metadata_dir = base_dir / metadata_dir

    reals = []
    for p in metadata_dir.glob("*.json"):
        m = load_metadata(p)
        if m.get("label_main") == "real_sync":
            reals.append(m)

    all_generated = []
    fakes_meta_dir = base_dir / config["data"].get("fakes_dir", "data/processed/fakes") / "metadata"
    fakes_meta_dir.mkdir(parents=True, exist_ok=True)

    for real in reals:
        others = [r for r in reals if r["sample_id"] != real["sample_id"]]
        fakes = generate_all_fakes_for_real(real, others, config, base_dir)
        for meta in fakes:
            save_metadata(meta, fakes_meta_dir / f"{meta['sample_id']}.json")
            all_generated.append(meta)

    return all_generated
