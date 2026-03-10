"""
Train/val/test split – konuşmacı ayrımlı (speaker-disjoint).
Aynı speaker_id sadece bir split'te yer alır.
"""
from __future__ import annotations

from pathlib import Path
import random

from src.utils.io import load_config, load_metadata, save_split, project_root


def collect_all_samples(metadata_dir: Path, fakes_metadata_dir: Path | None) -> list[dict]:
    """Tüm real + fake metadata'ları toplar."""
    samples = []
    for p in metadata_dir.glob("*.json"):
        samples.append(load_metadata(p))
    if fakes_metadata_dir and fakes_metadata_dir.exists():
        for p in fakes_metadata_dir.glob("*.json"):
            samples.append(load_metadata(p))
    return samples


def random_split(
    samples: list[dict],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Speaker bilgisi güvenilir değilse basit random split."""
    random.seed(seed)
    n = len(samples)
    if n == 0:
        return [], [], []
    idx = list(range(n))
    random.shuffle(idx)
    n_test = max(1, int(n * test_ratio))
    n_val = max(1, int(n * val_ratio))
    n_train = max(1, n - n_test - n_val)
    test_idx = set(idx[:n_test])
    val_idx = set(idx[n_test : n_test + n_val])
    train, val, test = [], [], []
    for i, s in enumerate(samples):
        if i in test_idx:
            test.append(s)
        elif i in val_idx:
            val.append(s)
        else:
            train.append(s)
    return train, val, test


def speaker_disjoint_split(
    samples: list[dict],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Önce speaker_id'lere göre grupla, sonra speaker'ları train/val/test'e böl.
    Böylece aynı konuşmacı birden fazla split'te olmaz.
    """
    random.seed(seed)
    by_speaker = {}
    for s in samples:
        spk = s.get("speaker_id", "unknown")
        by_speaker.setdefault(spk, []).append(s)

    speakers = list(by_speaker.keys())
    # Eğer tek konuşmacı varsa speaker-disjoint anlamlı değil; random split'e düş.
    if len(speakers) <= 1:
        return random_split(samples, val_ratio, test_ratio, seed)

    random.shuffle(speakers)
    n = len(speakers)
    n_test = max(1, int(n * test_ratio))
    n_val = max(1, int(n * val_ratio))
    n_train = n - n_test - n_val
    if n_train < 1:
        n_train, n_val, n_test = n - 2, 1, 1

    test_speakers = set(speakers[:n_test])
    val_speakers = set(speakers[n_test : n_test + n_val])
    train_speakers = set(speakers[n_test + n_val :])

    train = []
    val = []
    test = []
    for s in samples:
        spk = s.get("speaker_id", "unknown")
        if spk in test_speakers:
            test.append(s)
        elif spk in val_speakers:
            val.append(s)
        else:
            train.append(s)

    return train, val, test


def run_build_splits(config_path: str | Path | None = None) -> tuple[list[dict], list[dict], list[dict]]:
    """Config'teki metadata ve fakes dizinlerini kullanıp train/val/test yazar."""
    base = project_root()
    config = load_config(config_path or base / "configs" / "data.yaml")
    data = config.get("data", {})
    split_cfg = config.get("split", {})

    metadata_dir = base / data.get("metadata_dir", "data/processed/metadata")
    fakes_dir = base / data.get("fakes_dir", "data/processed/fakes")
    fakes_meta_dir = fakes_dir / "metadata"
    splits_dir = base / data.get("splits_dir", "data/splits")

    samples = collect_all_samples(metadata_dir, fakes_meta_dir)
    if not samples:
        return [], [], []

    train, val, test = speaker_disjoint_split(
        samples,
        val_ratio=split_cfg.get("val_ratio", 0.15),
        test_ratio=split_cfg.get("test_ratio", 0.15),
        seed=split_cfg.get("seed", 42),
    )

    save_split(train, splits_dir / "train.json")
    save_split(val, splits_dir / "val.json")
    save_split(test, splits_dir / "test.json")
    return train, val, test
