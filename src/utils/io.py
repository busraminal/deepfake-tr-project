"""Config, split ve metadata I/O."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(config: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, default_flow_style=False)


def load_json_splits(splits_dir: str | Path) -> tuple[list[dict], list[dict], list[dict]]:
    """train.json, val.json, test.json yükler."""
    d = Path(splits_dir)
    def load(name: str) -> list[dict]:
        import json
        p = d / f"{name}.json"
        if not p.exists():
            return []
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return load("train"), load("val"), load("test")


def save_split(samples: list[dict], path: str | Path) -> None:
    import json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)


def load_metadata(path: str | Path) -> dict[str, Any]:
    import json
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_metadata(meta: dict[str, Any], path: str | Path) -> None:
    import json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def project_root() -> Path:
    """Proje kökü (run_pipeline.py'nin olduğu dizin)."""
    return Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path, base: Path | None = None) -> Path:
    base = base or project_root()
    p = Path(path)
    if not p.is_absolute():
        p = base / p
    return p.resolve()
