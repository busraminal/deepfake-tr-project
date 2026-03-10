"""
LLM benchmark: metin embedding'leri ile dudak embedding'lerini karşılaştır.

Kullanım örneği:

  python scripts/llm_benchmark.py --split test --max-samples 800 \
    --out-md paper/llm_benchmark.md --out-tex paper/llm_benchmark.tex

Notlar:
- Simple encoder: mevcut TextEncoder (sync_model.pt içindeki "text" kısmı).
- Whisper+SBERT: opsiyonel; kütüphaneler yoksa bu satır hata vermeden atlanır.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import time

import numpy as np
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import load_config, load_json_splits, project_root
from src.evaluation.metrics import compute_all
from src.datasets.sync_dataset import SyncDataset, SyncTorchDataset
from src.models.mouth_model import MouthEncoder
from src.models.text_encoder import TextEncoder

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore


def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return (a * b).sum(axis=1)


def _embed_whisper_sbert(audio_paths, base_dir: Path) -> np.ndarray:
    """Whisper + SBERT embedding (varsa). Yoksa shape=(0,0) döner."""
    try:
        import whisper  # type: ignore
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError:
        print("Whisper veya sentence-transformers yok; whisper_sbert atlanacak.")
        return np.zeros((0, 0), dtype=np.float32)

    w_model = whisper.load_model("base")
    sbert = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    texts = []
    for rel in audio_paths:
        if not rel:
            texts.append("")
            continue
        path = base_dir / rel
        if not path.is_file():
            texts.append("")
            continue
        result = w_model.transcribe(str(path), language="tr")
        txt = (result.get("text") or "").strip()
        texts.append(txt)

    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    emb = sbert.encode(texts, convert_to_numpy=True)
    return emb.astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser(description="LLM benchmark: text vs lip embeddings")
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--max-samples", type=int, default=800)
    p.add_argument("--out-md", type=str, default=None)
    p.add_argument("--out-tex", type=str, default=None)
    args = p.parse_args()

    base = project_root()
    cfg = load_config(base / "configs" / "data.yaml")
    data_cfg = cfg.get("data", {})

    train, val, test = load_json_splits(base / data_cfg.get("splits_dir", "data/splits"))
    samples_all = {"train": train, "val": val, "test": test}[args.split]
    if not samples_all:
        print("No samples for split=%s" % args.split)
        return

    samples = samples_all[: args.max_samples]
    print("Using %d samples from split=%s" % (len(samples), args.split))

    if not TORCH_AVAILABLE:
        print("torch yok; simple encoder için de embedding üretilemez.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mouth_sz = cfg.get("preprocess", {}).get("mouth_size", 96)
    ds = SyncDataset(samples, base, mouth_size=(mouth_sz, mouth_sz))
    torch_ds = SyncTorchDataset(ds)
    loader = torch.utils.data.DataLoader(torch_ds, batch_size=8, shuffle=False, num_workers=0)

    # Sync checkpoint'ten mouth + text encoder'i çek
    sync_ckpt = base / "checkpoints" / "sync_model.pt"
    m_cfg = (load_config(base / "configs" / "train_sync.yaml") or {}).get("model", {})
    emb_dim = m_cfg.get("embedding_dim", 256)

    mouth_enc = MouthEncoder(embedding_dim=emb_dim)
    state = torch.load(sync_ckpt, map_location="cpu")
    mouth_enc.load_state_dict(state["mouth"], strict=True)
    mouth_enc.to(device).eval()

    text_enc = TextEncoder(vocab_size=1000, embedding_dim=emb_dim)
    text_enc.load_state_dict(state["text"], strict=True)
    text_enc.to(device).eval()

    y_true = []
    m_list = []
    t_simple_list = []
    audio_rel = []

    t0 = time()
    with torch.no_grad():
        for mouth, text, label in loader:
            mouth = mouth.to(device)
            text = text.to(device)
            m_emb = mouth_enc(mouth)          # (B, D)
            t_emb = text_enc(text)            # (B, D)
            m_list.append(m_emb.cpu().numpy())
            t_simple_list.append(t_emb.cpu().numpy())
            y_true.extend(label.numpy().tolist())
    m_arr = np.concatenate(m_list, axis=0)
    t_simple = np.concatenate(t_simple_list, axis=0)
    y_true_arr = np.asarray(y_true, dtype=np.float32)
    print("Simple encoder pass done in %.1fs, shape=%s" % (time() - t0, m_arr.shape))

    # Simple encoder skoru
    s_simple = _cosine(m_arr, t_simple)
    res = {"simple": {"metrics": compute_all(y_true_arr, s_simple, threshold=None)}}

    # Whisper+SBERT (isteğe bağlı)
    for s in samples:
        audio_rel.append(s.get("audio_path", ""))
    t_ws = _embed_whisper_sbert(audio_rel, base)
    if t_ws.shape[0] == m_arr.shape[0] and t_ws.size > 0:
        # Boyutlar farklı olabilir (örneğin SBERT 384, mouth 256).
        # Cosine için ortak boyutu min(D_mouth, D_text) olarak alıyoruz.
        d = min(m_arr.shape[1], t_ws.shape[1])
        m_proj = m_arr[:, :d]
        t_proj = t_ws[:, :d]
        s_ws = _cosine(m_proj, t_proj)
        res["whisper_sbert"] = {
            "metrics": compute_all(y_true_arr, s_ws, threshold=None),
        }

    print(json.dumps(res, indent=2, ensure_ascii=False))

    # Markdown tablo
    if args.out_md:
        lines = [
            "# LLM benchmark",
            "",
            "| Model | Accuracy | AUC | EER |",
            "|-------|----------|-----|-----|",
        ]
        for name, r in res.items():
            m = r["metrics"]
            lines.append(
                "| %s | %.3f | %.3f | %.3f |"
                % (name, m["accuracy"], m["auc"], m["eer"])
            )
        Path(args.out_md).write_text("\n".join(lines), encoding="utf-8")
        print("Written:", args.out_md)

    # LaTeX tablo
    if args.out_tex:
        tex = [
            "% Auto-generated by scripts/llm_benchmark.py",
            "\\begin{table}[htbp]",
            "  \\centering",
            "  \\caption{LLM benchmark results on %s split ($n=%d$).}" % (args.split, len(y_true_arr)),
            "  \\label{tab:llm_benchmark}",
            "  \\begin{tabular}{lccc}",
            "    \\toprule",
            "    Model & Acc. & AUC & EER \\\\",
            "    \\midrule",
        ]
        for name, r in res.items():
            m = r["metrics"]
            tex.append(
                "    %s & %.3f & %.3f & %.3f \\\\"
                % (name, m["accuracy"], m["auc"], m["eer"])
            )
        tex.extend(
            [
                "    \\bottomrule",
                "  \\end{tabular}",
                "\\end{table}",
            ]
        )
        Path(args.out_tex).write_text("\n".join(tex), encoding="utf-8")
        print("Written:", args.out_tex)


if __name__ == "__main__":
    main()

