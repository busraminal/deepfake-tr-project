"""Print dataset stats for the paper (splits, speakers)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
splits_dir = ROOT / "data" / "splits"
for name in ["train", "val", "test"]:
    p = splits_dir / f"{name}.json"
    if not p.exists():
        continue
    data = json.loads(p.read_text(encoding="utf-8"))
    speakers = set(s.get("speaker_id") for s in data)
    print(f"{name}: n={len(data)}, speakers={len(speakers)}")
total = 0
for name in ["train", "val", "test"]:
    p = splits_dir / f"{name}.json"
    if p.exists():
        total += len(json.loads(p.read_text(encoding="utf-8")))
print(f"total samples: {total}")
