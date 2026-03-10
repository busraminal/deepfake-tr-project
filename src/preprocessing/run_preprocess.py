"""
Tek bir ham video için uçtan uca ön işleme:
ses çıkarma, frame, yüz, ağız ROI, transkript ve metadata yazma.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.preprocessing.extract_audio import extract_audio
from src.preprocessing.extract_frames import extract_frames
from src.preprocessing.detect_face import crop_faces_from_frame_paths
from src.preprocessing.extract_mouth_roi import extract_mouth_rois_from_face_dir
from src.preprocessing.transcribe_tr import transcribe_tr
from src.utils.schema import sample_schema, LABEL_REAL_SYNC, MISMATCH_NONE
from src.utils.io import load_config, project_root, save_metadata


def get_duration_sec(video_path: Path) -> float:
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n / fps if fps > 0 else 0.0


def run_preprocess_one(
    video_path: str | Path,
    sample_id: str,
    speaker_id: str,
    config: dict[str, Any] | None = None,
    base_dir: Path | None = None,
    use_whisper: bool = True,
) -> dict[str, Any] | None:
    """
    Tek videoyu işler; metadata sözlüğü döner (real_sync örneği).
    """
    base_dir = base_dir or project_root()
    config = config or load_config(base_dir / "configs" / "data.yaml")
    data = config.get("data", {})
    preprocess = config.get("preprocess", {})

    raw_videos = base_dir / data.get("raw_videos_dir", "data/raw_videos")
    raw_audio_dir = base_dir / data.get("raw_audio_dir", "data/raw_audio")
    processed = base_dir / data.get("processed_dir", "data/processed")
    faces_dir = processed / "faces" / sample_id
    mouths_dir = processed / "mouths" / sample_id
    transcripts_dir = processed / "transcripts"
    metadata_dir = processed / "metadata"

    video_path = Path(video_path)
    if not video_path.is_absolute():
        video_path = raw_videos / video_path
    if not video_path.exists():
        return None

    fps = preprocess.get("fps", 25)
    face_size = preprocess.get("face_size", 224)
    mouth_size = preprocess.get("mouth_size", 96)
    audio_sr = preprocess.get("audio_sr", 16000)

    # 1) Ses
    raw_audio_dir.mkdir(parents=True, exist_ok=True)
    audio_path = raw_audio_dir / f"{sample_id}.wav"
    if not extract_audio(video_path, audio_path, sr=audio_sr):
        return None

    # 2) Frames (geçici klasör sample_id ile)
    frames_dir = processed / "frames" / sample_id
    frame_paths = extract_frames(video_path, frames_dir, fps=fps, prefix="frame")
    if not frame_paths:
        return None

    # 3) Yüz crop
    face_paths = crop_faces_from_frame_paths(frames_dir, faces_dir, size=(face_size, face_size), prefix="face")
    if not face_paths:
        return None

    # 4) Ağız ROI
    mouth_paths = extract_mouth_rois_from_face_dir(faces_dir, mouths_dir, size=(mouth_size, mouth_size), prefix="mouth")

    # 5) Transkript
    transcript = transcribe_tr(audio_path, use_whisper=use_whisper)
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    (transcripts_dir / f"{sample_id}.txt").write_text(transcript, encoding="utf-8")

    duration = get_duration_sec(video_path)
    def _rel(b: Path, d: Path) -> str:
        try:
            return str(b.resolve().relative_to(d.resolve()))
        except ValueError:
            return str(b)
    rel_video = _rel(video_path, base_dir)
    rel_audio = _rel(audio_path, base_dir)

    meta = sample_schema(
        sample_id=sample_id,
        speaker_id=speaker_id,
        video_path=rel_video,
        audio_path=rel_audio,
        transcript_tr=transcript,
        label_main=LABEL_REAL_SYNC,
        label_visual_fake=0,
        label_audio_fake=0,
        label_sync=1,
        sync_shift_ms=None,
        mismatch_type=MISMATCH_NONE,
        domain="academic",
        duration_sec=duration,
        frames_dir=_rel(frames_dir, base_dir),
        faces_dir=_rel(faces_dir, base_dir),
        mouths_dir=_rel(mouths_dir, base_dir),
    )
    metadata_dir.mkdir(parents=True, exist_ok=True)
    save_metadata(meta, metadata_dir / f"{sample_id}.json")
    return meta


def run_preprocess_all(
    video_list: list[tuple[str, str, str]],
    config_path: str | Path | None = None,
) -> list[dict]:
    """
    video_list: [(video_filename_or_path, sample_id, speaker_id), ...]
    """
    base = project_root()
    config = load_config(config_path or base / "configs" / "data.yaml")
    results = []
    for video_path, sample_id, speaker_id in video_list:
        out = run_preprocess_one(video_path, sample_id, speaker_id, config=config, base_dir=base)
        if out:
            results.append(out)
    return results
