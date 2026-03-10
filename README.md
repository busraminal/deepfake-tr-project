# Deepfake TR Project

**Multimodal speech–visual consistency analysis** — use case: deepfake / manipülasyon tespiti.

Proje, “sadece deepfake detector” değil; **konuşma + dudak hareketi + görüntü tutarlılığını** birlikte inceleyen bir sistem olarak konumlandırılır. Türkçe konuşma–dudak tutarlılığı odaklı kontrollü veri tasarımı ile makale hipotezini test etmeye uygundur.

Sistem üç bileşeni birlikte kullanır:
1. **Görüntü** gerçekten doğal mı? (S_v)
2. **Ses** gerçekten doğal mı?
3. **Konuşma ile dudak hareketi** uyumlu mu? (S_l)  
Final skor: **S_f = α·S_v + (1−α)·S_l**

Makale yol haritası ve deney tablosu: **[docs/PAPER_ROADMAP.md](docs/PAPER_ROADMAP.md)**

## Kurulum

**Önce proje klasörüne girin** (tüm komutlar bu dizinden çalıştırılmalı):

```bash
cd c:\Users\busra\Desktop\df_llm\deepfake_tr_project
pip install -r requirements.txt
```

Windows PowerShell'de tek satır: `cd C:\Users\busra\Desktop\df_llm\deepfake_tr_project`

Opsiyonel: `pip install soundfile`, `openai-whisper`, `ffmpeg` veya `moviepy`.

---

## Pipeline (attığın format — tek komut)

Proje, verdiğin formatta **uçtan uca çalışan tek pipeline** ile kuruldu:

```
  data/raw_videos  +  data/raw_audio
           ↓
  preprocessing (extract_audio, extract_frames, detect_face, extract_mouth_roi, transcribe_tr)
           ↓
  fake generation (sync_shift, content_mismatch, synthetic_audio)
           ↓
  data/splits (train.json, val.json, test.json) — speaker-disjoint
           ↓
  train visual (S_v) → train sync (S_l) → train fusion (S_f = α·S_v + (1−α)·S_l)
           ↓
  evaluate (accuracy, AUC, EER, F1)  →  paper/results_table.md
```

**Tek komutla tüm pipeline (demo veri ile):**

```bash
python run_pipeline.py full --demo
```

**Önemli:** Demo verideki yüz görselleri **tek renk placeholder**dır (gerçek yüz değil). **Tüm veriyi gerçek veriyle değiştirmek** için: **[docs/GERCEK_VERI_ADIMLAR.md](docs/GERCEK_VERI_ADIMLAR.md)** — Adım 1: `python scripts/clear_demo_data.py`, Adım 2: `add_real_videos.py avlips` veya `urls`, Adım 3–4: `generate_fakes`, `build_splits`.

**Daha büyük veri seti (100 real, ~700 fake):** `python run_pipeline.py full --demo --large` — 20 konuşmacı × 5 video = 100 real; sync ve fusion da eğitilir.

Bu komut sırayla: (1) demo veri, (2) sahte üretimi, (3) split, (4) görsel eğitim, (5) senkron eğitim, (6) fusion, (7) değerlendirme, (8) makale tablosu. Checkpoint’ler `checkpoints/` altına, sonuç tablosu `paper/results_table.md` dosyasına yazılır.

**Mevcut veri ile (preprocess zaten yapıldıysa):**

```bash
python run_pipeline.py full
```

Belirli adımları atlamak için: `--skip-fakes`, `--skip-splits`, `--skip-train-visual`, `--skip-train-sync`, `--skip-train-fusion`.

---

## Tek komutla demo (sadece visual)

Sadece demo veri + görsel model + değerlendirme (sync/fusion yok):

```bash
python run_pipeline.py demo
```

## Pipeline adımları (ayrı ayrı)

| Adım | Komut |
|------|--------|
| Ön işleme | `python run_pipeline.py preprocess --video-list video_list.csv` veya `--video / --sample-id / --speaker-id` |
| Sahte üretimi | `python run_pipeline.py generate_fakes` |
| Split (konuşmacı ayrımlı) | `python run_pipeline.py build_splits` |
| Eğitim (görsel) | `python run_pipeline.py train --model visual` |
| Eğitim (senkron) | `python run_pipeline.py train --model sync --config configs/train_sync.yaml` |
| Eğitim (füzyon) | `python run_pipeline.py train --model fusion --config configs/fusion.yaml` |
| Değerlendirme | `python run_pipeline.py evaluate` (accuracy, AUC, EER, F1; visual + varsa sync/fusion) |
| **Tüm modeller tek seferde** | `python run_pipeline.py experiments --split test --out paper/results.json` (Visual + Sync + Fusion + tablo) |

## Gerçek videoları veri setine ekleme

**Tek script ile gerçek videoları ekleyebilirsin:**

1. **AVLips (İngilizce lip-sync real+fake, ~9 GB):**  
   `python scripts/add_real_videos.py avlips --download`  
   (İndirir; zip’i açtıktan sonra aynı komutu `--download` olmadan çalıştırırsan dönüştürme + yüz/ağız ön işleme yapar.)  
   Sonra: `python run_pipeline.py generate_fakes` ve `python run_pipeline.py build_splits`.

2. **URL listesi (Türkçe vb.):**  
   `data/video_urls_example.csv` gibi `url,sample_id,speaker_id` CSV hazırla; placeholder URL’leri kendi linklerinle değiştir.  
   `python scripts/add_real_videos.py urls --csv data/video_urls.csv`  
   (yt-dlp ile indirir, ön işlem yapar; ardından `generate_fakes` + `build_splits`.)

Daha fazla veri seti: **[docs/DATASETS.md](docs/DATASETS.md)**.

## Veri seti şeması

- **Sınıflar:** `real_sync`, `fake_sync_shift`, `fake_content_mismatch`, `fake_audio_synthetic`, (opsiyonel) `fake_visual`
- **Etiketler:** `label_visual_fake`, `label_audio_fake`, `label_sync`, `sync_shift_ms`, `mismatch_type`
- Ayrıntılı şema ve protokol: **[DATA_SPEC.md](DATA_SPEC.md)**

## Tek örnek için skor (inference)

Eğitilmiş görsel model ile bir sample'ın S_v skorunu hesaplamak için (metadata + faces mevcut olmalı):

```bash
python -m src.inference.predict_video --sample-id demo_001
```

Çıktı: `S_v (görsel sahte olasılığı): 0.xx  (0=gerçek, 1=sahte)`

## Makale tablosu (deney sonuçları)

Tüm modelleri **tek komutla** test edip hem konsol hem tablo + JSON almak için:

```bash
python run_pipeline.py experiments --split test --out paper/results.json
```

(Sırayla Visual, Sync, Fusion değerlendirilir; `paper/results_table.md` ve isteğe bağlı `paper/results.json` oluşturulur.)

Sadece konsol çıktısı için:

```bash
python run_pipeline.py evaluate
# veya tablo dosyaya:
python scripts/export_results_table.py --split test --out paper/results_table.md
```

Makaledeki LaTeX tablosunu guncellemek icin: `python scripts/export_results_latex.py --split test --out paper/results_table.tex`  
Fusion $\alpha$ ablasyonu: `python scripts/run_ablation_alpha.py --out paper/ablation_alpha.tex`

Çıktı: **Model | Accuracy | AUC | EER | F1** (Visual only, Sync only, Fusion). Karar eşiği ROC Youden (argmax(TPR−FPR)) ile seçilir; sabit 0.5 kullanılmaz. Skor dağılımı teşhisi: `python scripts/plot_score_histograms.py --out paper/figures/score_hist.png`

## Projeyi güçlendirme

- **Metrikler:** `evaluate` test setinde accuracy, AUC, EER, F1, precision, recall raporlar.
- **Sync + fusion:** `train --model sync` ve `train --model fusion` sonrası `evaluate` üç kolun sonuçlarını verir.
- **Gerçek veri:** [docs/DATASETS.md](docs/DATASETS.md) (AVLips, FakeAVCeleb, LRS3, vb.); makale için 50–100+ real, 200–400 fake hedeflenir.
- **Türkçe iddia:** Gerçek Türkçe cümleler + transkript ile aynı pipeline kullanılır.
- **Konumlandırma ve adımlar:** [docs/PAPER_ROADMAP.md](docs/PAPER_ROADMAP.md).

## Yapı

- **data/** – raw_videos, raw_audio, processed (faces, mouths, transcripts, metadata), fakes, splits
- **configs/** – data.yaml, train_visual.yaml, train_sync.yaml, fusion.yaml
- **src/** – preprocessing, datasets, models, training, inference, **evaluation** (metrics, evaluate_visual/sync/fusion), utils
- **notebooks/** – EDA, error analysis
- **paper/** – figures, main.tex
