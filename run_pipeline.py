"""
Ana pipeline — attığın format, uçtan uca:

  raw_videos / raw_audio
       ↓
  preprocessing (extract_audio, frames, face, mouth, transcript)
       ↓
  fake generation (sync_shift, content_mismatch, synthetic_audio)
       ↓
  dataset split (train/val/test, speaker-disjoint)
       ↓
  train visual → train sync → train fusion
       ↓
  evaluate + export results table

Proje kökünden: python run_pipeline.py full --demo   (tek komut, demo veri ile)
                python run_pipeline.py full         (mevcut veri ile, preprocess atlanır)
"""
import argparse
import sys
import time
from pathlib import Path

# Proje kökünü path'e ekle
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _step(name: str):
    """Adım başlığı ve süre ölçümü."""
    t0 = time.time()
    print(f"\n{'='*60}\n  {name}\n{'='*60}")
    return t0


def _step_done(t0):
    print(f"  -> Tamamlandi ({time.time() - t0:.1f}s)\n")


def cmd_preprocess(args):
    """Ham videoları işler (liste dosyadan veya tek video)."""
    from src.utils.io import load_config, project_root
    from src.preprocessing.run_preprocess import run_preprocess_one, run_preprocess_all
    base = project_root()
    config = load_config(args.config)
    if args.video_list:
        # video_list: her satır "video_path,sample_id,speaker_id"
        lines = Path(args.video_list).read_text(encoding="utf-8").strip().splitlines()
        video_list = []
        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                video_list.append((parts[0], parts[1], parts[2]))
        results = run_preprocess_all(video_list, args.config)
        print(f"Processed {len(results)} videos.")
    elif args.video and args.sample_id and args.speaker_id:
        meta = run_preprocess_one(
            args.video, args.sample_id, args.speaker_id,
            config=config, base_dir=base, use_whisper=args.whisper,
        )
        if meta:
            print("OK:", meta["sample_id"])
        else:
            print("Preprocess failed.")
    else:
        print("Use --video-list FILE or --video / --sample-id / --speaker-id")
        sys.exit(1)


def cmd_generate_fakes(args):
    """Real metadata'lardan sahte örnekler üretir."""
    from src.utils.io import load_config, project_root
    from src.preprocessing.generate_fakes import run_generate_fakes
    base = project_root()
    config_path = base / args.config if not Path(args.config).is_absolute() else Path(args.config)
    config = load_config(config_path)
    metadata_dir = base / config.get("data", {}).get("metadata_dir", "data/processed/metadata")
    generated = run_generate_fakes(metadata_dir, config_path)
    print(f"Generated {len(generated)} fake samples.")


def cmd_build_splits(args):
    """Konuşmacı ayrımlı train/val/test split oluşturur."""
    from src.utils.io import project_root
    from src.preprocessing.build_splits import run_build_splits
    base = project_root()
    config_path = base / args.config if not Path(args.config).is_absolute() else Path(args.config)
    train, val, test = run_build_splits(config_path)
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")


def cmd_train(args):
    """Eğitim: visual, sync veya fusion."""
    from src.utils.io import load_config, project_root
    base = project_root()
    if args.model == "visual":
        from src.training.train_visual import main
        main(base / args.config)
    elif args.model == "sync":
        from src.training.train_sync import main
        main(base / args.config)
    elif args.model == "fusion":
        from src.training.train_fusion import main
        main(base / args.config)
    else:
        print("--model must be visual | sync | fusion")
        sys.exit(1)


def cmd_full(args):
    """
    Uçtan uca pipeline (attığın format):
    [demo veri] → generate_fakes → build_splits → train visual → train sync → train fusion → evaluate → export table
    """
    from src.utils.io import project_root, load_config
    base = project_root()
    config_path = base / getattr(args, "config", "configs/data.yaml")
    config = load_config(config_path)
    data_cfg = config.get("data", {})

    if getattr(args, "demo", False):
        t0 = _step("1. Demo veri olusturma")
        from scripts.create_demo_data import create_demo_data
        if getattr(args, "large", False):
            create_demo_data(base_dir=base, num_speakers=20, samples_per_speaker=5)
        else:
            create_demo_data(base_dir=base, num_speakers=3, samples_per_speaker=2)
        _step_done(t0)

    if not getattr(args, "skip_fakes", False):
        t0 = _step("2. Sahte örnek üretimi (sync_shift, content_mismatch, synthetic_audio)")
        from src.preprocessing.generate_fakes import run_generate_fakes
        metadata_dir = base / data_cfg.get("metadata_dir", "data/processed/metadata")
        if metadata_dir.exists():
            generated = run_generate_fakes(metadata_dir, config_path)
            print(f"  -> {len(generated)} sahte ornek uretildi.")
        else:
            print("  -> Metadata yok; once demo (--demo) veya preprocess calistir.")
        _step_done(t0)
    else:
        print("\n  [2. Sahte üretimi atlandı (--skip-fakes)]")

    if not getattr(args, "skip_splits", False):
        t0 = _step("3. Split (train/val/test, speaker-disjoint)")
        from src.preprocessing.build_splits import run_build_splits
        train, val, test = run_build_splits(config_path)
        print(f"  -> train: {len(train)}, val: {len(val)}, test: {len(test)}")
        _step_done(t0)
    else:
        print("\n  [3. Split atlandı (--skip-splits)]")

    if not getattr(args, "skip_train_visual", False):
        t0 = _step("4. Görsel model eğitimi (S_v)")
        from src.training.train_visual import main as train_visual_main
        use_demo_cfg = getattr(args, "demo", False) and not getattr(args, "full_config", False)
        cfg = "configs/train_visual_demo.yaml" if use_demo_cfg else "configs/train_visual.yaml"
        train_visual_main(base / cfg)
        _step_done(t0)
    else:
        print("\n  [4. Görsel eğitim atlandı (--skip-train-visual)]")

    if not getattr(args, "skip_train_sync", False):
        t0 = _step("5. Senkron model eğitimi (S_l)")
        from src.training.train_sync import main as train_sync_main
        use_demo_cfg = getattr(args, "demo", False) and not getattr(args, "full_config", False)
        cfg = "configs/train_sync_demo.yaml" if use_demo_cfg else "configs/train_sync.yaml"
        train_sync_main(base / cfg)
        _step_done(t0)
    else:
        print("\n  [5. Senkron eğitim atlandı (--skip-train-sync)]")

    if not getattr(args, "skip_train_fusion", False):
        t0 = _step("6. Fusion model (S_f = α·S_v + (1−α)·S_l)")
        from src.training.train_fusion import main as train_fusion_main
        train_fusion_main(base / "configs/fusion.yaml")
        _step_done(t0)
    else:
        print("\n  [6. Fusion atlandı (--skip-train-fusion)]")

    t0 = _step("7. Değerlendirme (accuracy, AUC, EER, F1)")
    cmd_evaluate(args)
    _step_done(t0)

    t0 = _step("8. Makale tablosu (paper/results_table.md)")
    out = base / "paper" / "results_table.md"
    from scripts.export_results_table import main as export_main
    # export_main uses argparse; geçici olarak sys.argv ile --out ver
    _argv = sys.argv
    sys.argv = ["export_results_table.py", "--split", "test", "--out", str(out)]
    try:
        export_main()
    finally:
        sys.argv = _argv
    print(f"  -> {out}")
    _step_done(t0)

    print("\n" + "="*60)
    print("  PIPELINE TAMAMLANDI")
    print("  Checkpoints: checkpoints/visual_model.pt, sync_model.pt, fusion_model.pt")
    print("  Sonuç tablosu: paper/results_table.md")
    print("="*60 + "\n")


def cmd_demo(args):
    """Demo: create_demo_data -> generate_fakes -> build_splits -> train visual (2 epoch) -> evaluate."""
    from src.utils.io import project_root
    base = project_root()
    config_path = base / getattr(args, "config", "configs/data.yaml")
    print("1/5 Demo veri oluşturuluyor...")
    from scripts.create_demo_data import create_demo_data
    create_demo_data(base_dir=base, num_speakers=3, samples_per_speaker=2)
    print("2/5 Sahte örnekler üretiliyor...")
    from src.preprocessing.generate_fakes import run_generate_fakes
    from src.utils.io import load_config
    config = load_config(config_path)
    metadata_dir = base / config.get("data", {}).get("metadata_dir", "data/processed/metadata")
    run_generate_fakes(metadata_dir, config_path)
    print("3/5 Split oluşturuluyor...")
    from src.preprocessing.build_splits import run_build_splits
    run_build_splits(config_path)
    print("4/5 Görsel model eğitimi (2 epoch)...")
    from src.training.train_visual import main
    main(base / "configs" / "train_visual_demo.yaml")
    print("5/5 Test seti değerlendirmesi...")
    cmd_evaluate(args)
    print("Demo bitti. checkpoints/visual_model.pt kaydedildi.")


def cmd_evaluate(args):
    """Test seti üzerinde visual (ve varsa sync/fusion) metrikleri hesaplar."""
    from src.evaluation.evaluate_visual import run_evaluate_visual
    from src.evaluation.evaluate_sync import run_evaluate_sync
    from src.evaluation.evaluate_fusion import run_evaluate_fusion
    split = getattr(args, "split", "test")
    print("=== Değerlendirme (test) ===\n")
    res_vis = run_evaluate_visual(split=split)
    if "error" in res_vis:
        print("Visual:", res_vis["error"], res_vis.get("path", ""))
    else:
        print("Visual model:")
        print(f"  accuracy: {res_vis['accuracy']:.4f}  AUC: {res_vis['auc']:.4f}  EER: {res_vis['eer']:.4f}  F1: {res_vis['f1']:.4f}  (n={res_vis['n_samples']})")
    res_sync = run_evaluate_sync(split=split)
    if "error" in res_sync:
        print("Sync:   ", res_sync.get("error"), res_sync.get("path", "") or "(sync_model.pt yok - train --model sync ile eğit)")
    else:
        print("Sync model:")
        print(f"  accuracy: {res_sync['accuracy']:.4f}  AUC: {res_sync['auc']:.4f}  EER: {res_sync['eer']:.4f}  (n={res_sync['n_samples']})")
    res_fus = run_evaluate_fusion(split=split)
    if "error" in res_fus:
        if res_fus.get("error") == "sync_checkpoint_not_found":
            print("Fusion: sync checkpoint yok, atlandı.")
        else:
            print("Fusion:", res_fus.get("error"))
    else:
        print("Fusion (S_f = 0.5*S_v + 0.5*S_l):")
        print(f"  accuracy: {res_fus['accuracy']:.4f}  AUC: {res_fus['auc']:.4f}  EER: {res_fus['eer']:.4f}")
    print()


def cmd_experiments(args):
    """Tum modelleri (visual, sync, fusion) tek calistirmada test eder; tablo + istege bagli JSON."""
    from scripts.run_experiments import main as run_exp_main
    run_exp_main(split=getattr(args, "split", "test"), out=getattr(args, "out", None))


def main():
    parser = argparse.ArgumentParser(description="Deepfake TR pipeline")
    parser.add_argument("--config", type=str, default="configs/data.yaml", help="Data config path")
    sub = parser.add_subparsers(dest="step", help="Pipeline step")

    # preprocess
    p_pre = sub.add_parser("preprocess", help="Extract audio, frames, faces, mouths, transcript")
    p_pre.add_argument("--video-list", type=str, help="CSV file: video_path,sample_id,speaker_id per line")
    p_pre.add_argument("--video", type=str, help="Single video path")
    p_pre.add_argument("--sample-id", type=str, help="Sample ID (e.g. TRDF_00001)")
    p_pre.add_argument("--speaker-id", type=str, help="Speaker ID (e.g. spk_01)")
    p_pre.add_argument("--whisper", action="store_true", default=True, help="Use Whisper for transcription")
    p_pre.add_argument("--no-whisper", action="store_false", dest="whisper")
    p_pre.set_defaults(run=cmd_preprocess)

    # generate fakes
    p_fake = sub.add_parser("generate_fakes", help="Create fake_sync_shift, content_mismatch, synthetic")
    p_fake.set_defaults(run=cmd_generate_fakes)

    # build splits
    p_split = sub.add_parser("build_splits", help="Speaker-disjoint train/val/test")
    p_split.set_defaults(run=cmd_build_splits)

    # train
    p_train = sub.add_parser("train", help="Train visual, sync or fusion model")
    p_train.add_argument("--model", type=str, choices=["visual", "sync", "fusion"], required=True)
    p_train.add_argument("--config", type=str, default="configs/train_visual.yaml",
                         help="Train config (train_visual.yaml, train_sync.yaml, fusion.yaml)")
    p_train.set_defaults(run=cmd_train)

    # evaluate
    p_eval = sub.add_parser("evaluate", help="Run evaluation on test set (accuracy, AUC, EER, F1)")
    p_eval.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p_eval.set_defaults(run=cmd_evaluate)

    # experiments: tum modelleri tek seferde test et, JSON + paper tablosu uret
    p_exp = sub.add_parser("experiments", help="Visual + Sync + Fusion hepsini test et; tablo ve JSON cikti")
    p_exp.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p_exp.add_argument("--out", type=str, default=None, help="JSON cikti (ornegin paper/results.json)")
    p_exp.set_defaults(run=cmd_experiments)

    # demo: veri üret + fakes + split + 2 epoch visual train
    p_demo = sub.add_parser("demo", help="Demo veri oluştur, pipeline çalıştır (2 epoch visual)")
    p_demo.set_defaults(run=cmd_demo)

    # full: attığın format, uçtan uca (veri → fakes → split → train 3 model → evaluate → tablo)
    p_full = sub.add_parser("full", help="Tüm pipeline (attığın format); --demo ile demo veri kullan")
    p_full.add_argument("--demo", action="store_true", help="Demo veri olustur, sonra tum adimlar")
    p_full.add_argument("--large", action="store_true", help="Buyuk demo: 20 konusmaci x 5 video = 100 real, ~700 fake")
    p_full.add_argument("--full-config", action="store_true", help="Tam config (Visual 30, Sync 40 epoch); --demo ile de kullanilir")
    p_full.add_argument("--skip-fakes", action="store_true", help="Sahte uretimini atla")
    p_full.add_argument("--skip-splits", action="store_true", help="Split atla")
    p_full.add_argument("--skip-train-visual", action="store_true", help="Görsel eğitim atla")
    p_full.add_argument("--skip-train-sync", action="store_true", help="Senkron eğitim atla")
    p_full.add_argument("--skip-train-fusion", action="store_true", help="Fusion atla")
    p_full.set_defaults(run=cmd_full)

    args = parser.parse_args()
    if not args.step:
        parser.print_help()
        return
    if args.step == "train":
        args.config = getattr(args, "config", "configs/train_visual.yaml")
    if args.step == "evaluate":
        from src.utils.io import load_config
    args.run(args)


if __name__ == "__main__":
    main()
