# Makale Yol Haritası — Multimodal Speech–Visual Consistency Analysis

Bu proje **deepfake detection** olarak değil şu çerçevede konumlandırılır:

> **Multimodal speech–visual consistency analysis**  
> Use case: deepfake / manipülasyon tespiti.

Bu çerçeve, **konuşma + dudak hareketi + görüntü tutarlılığı** üzerine odaklandığı için ICASSP, INTERSPEECH, CVPR workshop gibi alanlarda daha net bir katkı olarak anlatılabilir.

---

## Mevcut Durum (durum analizi ile uyumlu)

| Bileşen | Durum | Not |
|--------|--------|-----|
| Pipeline (raw → preprocess → fakes → split → train → eval) | ✅ Çalışıyor | Akademik projede en zor kısım |
| Veri şeması (real_sync, sync_shift, content_mismatch, synthetic_audio) | ✅ Doğru | Makale hipotezini test etmeye uygun |
| Speaker-disjoint split (train/val/test) | ✅ Var | Kişi ezberi engellenir, gerçek performans ölçülür |
| Görsel model (S_v) | ✅ Eğitiliyor | face frames → CNN → S_v ∈ [0,1] |
| **Metrikler** (accuracy, AUC, EER, F1, precision, recall) | ✅ Hesaplanıyor | `python run_pipeline.py evaluate`; makale tablosu: `scripts/export_results_table.py` |
| Senkron model (S_l) | ⏳ Kod var, eğitim yapılacak | mouth ROI + transcript → S_l |
| Fusion (S_f = α·S_v + (1−α)·S_l) | ⏳ Kod var, S_v+S_l ile değerlendirme var | α sabit/öğrenilebilir |
| Gerçek / Türkçe veri | ❌ Eksik | Demo: 6 real, ~48 fake; makale için 50–100+ real, Türkçe cümleler |

---

## Makale Seviyesine Çıkarma Adımları

### Adım 1 — Gerçek veri

- **Hedef:** En az 50–100 gerçek video, 200–400 fake (kontrollü manipülasyon).
- **Kaynaklar:** LRS3, AVSpeech, FakeAVCeleb, AVLips; veya kendi çektiğin Türkçe videolar.
- **Türkçe iddia için:** Gerçek Türkçe cümleler + transkript (Whisper veya manuel).

### Adım 2 — Sync modeli eğit

```bash
python run_pipeline.py train --model sync --config configs/train_sync.yaml
```

- **Pipeline:** mouth ROI → temporal encoder → M_video; transcript → text encoder → E_text; S_l = 1 − cos(E_text, M_video).
- **Çıktı:** `checkpoints/sync_model.pt`; sonra `evaluate` ile sync metrikleri.

### Adım 3 — Fusion

- S_f = α·S_v + (1−α)·S_l (config’te α; isteğe göre öğrenilebilir).
- `evaluate` zaten visual + sync checkpoint’leri varsa fusion skorunu da raporlar.

### Dataset genişletme (gerçek veri)

- **Türkçe cümle havuzu:** `data/sentence_pool_tr.txt` (örnek cümleler; veri toplama protokolünde kullanılır).
- **Hedef:** 20 konuşmacı, 100 video, 300 fake, ~400 toplam.
- **Script:** `python scripts/dataset_expansion.py protocol` → `data/video_list_protocol.csv` şablonu (videoları bu yollara koyup preprocess çalıştır).
- **Doğrulama:** `python scripts/dataset_expansion.py validate` → hedef vs mevcut sayılar.

### Adım 4 — Deney tablosu (makale tablosu)

Tüm modeller eğitildikten sonra:

```bash
python run_pipeline.py evaluate
```

veya

```bash
python scripts/export_results_table.py
```

ile **model | accuracy | AUC | EER | F1** tablosunu al. Bu tablo makaledeki “main results” tablosu olabilir.

---

## Önerilen Makale Tablo Şablonu

| Model | Accuracy | Precision | Recall | F1 | AUC | EER |
|-------|----------|-----------|--------|-----|-----|-----|
| Visual only (S_v) | — | — | — | — | — | — |
| Sync only (S_l) | — | — | — | — | — | — |
| Fusion (S_f) | — | — | — | — | — | — |

Değerler `python run_pipeline.py evaluate` veya `scripts/export_results_table.py` / `scripts/run_experiments.py` ile doldurulur.

## Mimari figür

- **LaTeX (TikZ):** `paper/figures/architecture.tex` → `pdflatex architecture.tex` ile PDF.
- **PNG:** `python scripts/plot_architecture.py` → `paper/figures/architecture.png` (makalede `\includegraphics{figures/architecture.png}`).

## Makale katkıları (paper/main.tex)

1. Türkçe konuşma–dudak tutarlılığı veri seti  
2. Multimodal deepfake detection mimarisi  
3. Lip-sync uyumsuzluk metriği (S_l)  
4. Multimodal model karşılaştırması (visual, sync, fusion)

---

## Konumlandırma Özeti

- **Ana konu:** Multimodal speech–visual consistency (konuşma, dudak hareketi, görüntü tutarlılığı).
- **Use case:** Deepfake / manipülasyon tespiti; kontrollü manipülasyonlar (temporal shift, content mismatch, synthetic audio) ile deney.
- **Katkı:** Kontrollü veri tasarımı + speaker-disjoint değerlendirme + S_v, S_l, S_f ile tutarlılık analizi.

Bu çerçeve ile proje, sadece “bir deepfake detector” değil, **tutarlılık odaklı multimodal analiz** olarak anlatılabilir; makale potansiyeli bu şekilde daha net ortaya çıkar.
