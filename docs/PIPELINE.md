# Pipeline — Attığın Format, Uçtan Uca

Bu doküman, projenin **gerçekten çalışan** pipeline formatını özetler.

---

## Dizin Yapısı (format)

```
deepfake_tr_project/
├── data/
│   ├── raw_videos/
│   ├── raw_audio/
│   ├── processed/
│   │   ├── faces/
│   │   ├── mouths/
│   │   ├── transcripts/
│   │   ├── metadata/
│   │   └── fakes/
│   │       ├── audio/
│   │       └── metadata/
│   └── splits/
│       ├── train.json
│       ├── val.json
│       └── test.json
├── configs/
│   ├── data.yaml
│   ├── train_visual.yaml
│   ├── train_sync.yaml
│   └── fusion.yaml
├── src/
│   ├── preprocessing/   (extract_audio, extract_frames, detect_face, extract_mouth_roi, transcribe_tr)
│   ├── datasets/        (visual_dataset, sync_dataset, fusion_dataset)
│   ├── models/          (visual_model, mouth_model, text_encoder, fusion_model)
│   ├── training/        (train_visual, train_sync, train_fusion)
│   ├── inference/       (predict_video)
│   ├── evaluation/      (metrics, evaluate_visual, evaluate_sync, evaluate_fusion)
│   └── utils/           (io, schema, seed, logger)
├── scripts/             (create_demo_data, download_dataset, convert_foreign_dataset, export_results_table)
├── run_pipeline.py      (tek giriş noktası)
└── paper/
    └── results_table.md (makale tablosu)
```

---

## Akış (sıra)

| # | Adım | Çıktı |
|---|------|--------|
| 1 | **Preprocessing** | raw_videos → raw_audio, processed/frames, faces, mouths, transcripts, metadata |
| 2 | **Fake generation** | metadata (real_sync) → fakes/audio + fakes/metadata (sync_shift, content_mismatch, synthetic_audio) |
| 3 | **Split** | metadata + fakes/metadata → train.json, val.json, test.json (speaker-disjoint) |
| 4 | **Train visual** | train → S_v model → checkpoints/visual_model.pt |
| 5 | **Train sync** | train → S_l model → checkpoints/sync_model.pt |
| 6 | **Train fusion** | S_f = α·S_v + (1−α)·S_l → checkpoints/fusion_model.pt |
| 7 | **Evaluate** | test set → accuracy, AUC, EER, F1 (visual, sync, fusion) |
| 8 | **Export table** | paper/results_table.md (Model | Accuracy | AUC | EER | F1) |

---

## Tek Komut

```bash
# Tüm pipeline (demo veri ile)
python run_pipeline.py full --demo

# Mevcut veri ile (preprocess daha önce yapıldıysa)
python run_pipeline.py full

# Adım atlama (örn. sadece sync+fusion yeniden eğit)
python run_pipeline.py full --skip-fakes --skip-splits --skip-train-visual
```

---

## Sınıflar (veri formatı)

| label_main | Amaç |
|------------|------|
| real_sync | Referans (gerçek video + ses + senkron) |
| fake_sync_shift | Lip-sync hatası (ses zamanda kaydırıldı) |
| fake_content_mismatch | Konuşma–dudak uyumsuzluğu |
| fake_audio_synthetic | Sentetik ses |

Metadata alanları: `sample_id`, `speaker_id`, `video_path`, `audio_path`, `transcript_tr`, `label_main`, `label_visual_fake`, `label_audio_fake`, `label_sync`, `sync_shift_ms`, `mismatch_type`, `faces_dir`, `mouths_dir`, vb.

---

## Config Özeti

- **configs/data.yaml:** Yollar, hedefler, split oranları, ön işleme (fps, face_size, mouth_size, audio_sr).
- **configs/train_visual.yaml / train_sync.yaml / fusion.yaml:** Model ve eğitim parametreleri.
- Demo (hızlı) config’ler: `train_visual_demo.yaml`, `train_sync_demo.yaml`.

Bu format ile pipeline tek komutla (`full --demo` veya `full`) uçtan uca çalışır.
