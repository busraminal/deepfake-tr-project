# Tüm Veri Setini Gerçek Veri ile Değiştirme — Adım Adım

Demo (placeholder) veriyi kaldırıp yerine gerçek videolardan üretilmiş veri koymak için aşağıdaki adımları sırayla uygulayın.

---

## Adım 1: Demo veriyi temizle

Mevcut demo metadata, yüz/agiz görselleri ve ses dosyalarini siler. Fake örnekler (demo_* kaynakli) de silinir.

```bash
cd deepfake_tr_project
python scripts/clear_demo_data.py
```

**Ne silinir:** `data/processed/metadata/demo_*.json`, `data/processed/fakes/metadata/demo_*.*.json`, `data/processed/faces/demo_*`, `data/processed/mouths/demo_*`, `data/raw_audio/demo_*.wav`, ilgili fake ses dosyalari.

---

## Adım 2: Gerçek videolari ekle (iki seçenekten biri)

### Seçenek A — AVLips (hazir veri seti, ~9 GB)

```bash
# Indir (zip ~9 GB)
python scripts/add_real_videos.py avlips --download

# Zip'i ac: 7z x data/raw_videos/avlips/AVLips.zip  (veya Windows'ta sag tik > Birlestir)
# Sonra donustur + yuz/agiz cikar:
python scripts/add_real_videos.py avlips
```

### Seçenek B — URL listesi (YouTube, TED vb.)

1. `data/video_urls.csv` olustur; icerik ornegi:

   ```text
   url,sample_id,speaker_id
   https://www.youtube.com/watch?v=XXXXX,tr_001,spk_1
   https://www.youtube.com/watch?v=YYYYY,tr_002,spk_1
   ```

2. Indir + on isle:

   ```bash
   pip install yt-dlp
   python scripts/add_real_videos.py urls --csv data/video_urls.csv
   ```

---

## Adım 3: Sahte örnekleri üret

Gerçek örneklerden sync_shift, content_mismatch, synthetic_audio fake'leri uretilir.

```bash
python run_pipeline.py generate_fakes
```

---

## Adım 4: Train/val/test böl

Konusmaci ayrımlı (speaker-disjoint) split.

```bash
python run_pipeline.py build_splits
```

---

## Adım 5: Istatistikleri kontrol et

```bash
python scripts/paper_stats.py
```

Cikti: train/val/test ornek ve konusmaci sayilari. Makul (örn. train 200+, test 50+) olana kadar Adim 2’de daha fazla video ekleyebilirsiniz.

---

## Adım 6: Modelleri eğit (isteğe bağlı)

Gerçek veri ile yeniden eğitim:

```bash
python run_pipeline.py train --model visual --config configs/train_visual.yaml
python run_pipeline.py train --model sync --config configs/train_sync.yaml
python run_pipeline.py train --model fusion --config configs/fusion.yaml
```

Veya tek komutla (veri zaten hazirsa):

```bash
python run_pipeline.py full --skip-fakes --skip-splits
```

(Önce Adim 3 ve 4 yapildiysa --skip-fakes ve --skip-splits kullanilir; aksi halde full komutunda fakes ve splits tekrar calisir.)

---

## Adım 7: Sonuç tablolarını güncelle

```bash
python scripts/export_results_latex.py --split test --out paper/results_table.tex
python scripts/run_ablation_alpha.py --split test --out paper/ablation_alpha.tex
```

---

## Özet sıra

| # | Adım | Komut |
|---|------|--------|
| 1 | Demo temizle | `python scripts/clear_demo_data.py` |
| 2 | Gerçek veri ekle | `add_real_videos.py avlips` veya `urls --csv ...` |
| 3 | Fake üret | `python run_pipeline.py generate_fakes` |
| 4 | Split | `python run_pipeline.py build_splits` |
| 5 | Kontrol | `python scripts/paper_stats.py` |
| 6 | Eğit | `train --model visual/sync/fusion` veya `full --skip-fakes --skip-splits` |
| 7 | Tablolar | `export_results_latex.py` + `run_ablation_alpha.py` |
