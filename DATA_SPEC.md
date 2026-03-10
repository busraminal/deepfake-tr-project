# Veri Seti Spesifikasyonu — Deepfake TR

Bu belge, **Türkçe konuşma–dudak tutarlılığı odaklı kontrollü multimodal deepfake benchmark** veri setinin yapısını, etiket şemasını ve üretim sürecini tanımlar.

---

## 1. Amaç

Veri seti yalnızca “video toplamak” için değil; aşağıdaki soruları yanıtlayacak **kontrollü deney zemini** kurmak içindir:

- **Soru A:** Görüntü olarak video sahte mi?
- **Soru B:** Ses olarak sahte mi?
- **Soru C:** Ses ve dudak hareketi senkron ve anlamsal olarak tutarlı mı?
- **Soru D:** Görsel + senkron bilgisi birleşince daha iyi tespit yapılabiliyor mu?

Makaledeki skorlar:

- \( S_v \) = görsel sahtecilik skoru  
- \( S_l \) = konuşma–dudak uyumsuzluk skoru  
- \( S_f = \alpha S_v + (1-\alpha) S_l \) = final skor  

Veri seti, \( S_v \), \( S_l \) ve \( S_f \) modellerini besleyecek şekilde tasarlanmıştır.

---

## 2. Klasör ve Dosya Yapısı

```
data/
├── raw_videos/          # Ham videolar (gerçek kayıtlar)
├── raw_audio/           # Videolardan çıkarılan WAV
├── processed/
│   ├── frames/          # Örnek bazlı frame klasörleri (sample_id)
│   ├── faces/           # Yüz crop'ları (sample_id)
│   ├── mouths/           # Ağız ROI'leri (sample_id)
│   ├── transcripts/     # sample_id.txt
│   ├── metadata/        # sample_id.json (gerçek örnekler)
│   └── fakes/
│       ├── audio/       # Üretilmiş sahte sesler
│       └── metadata/    # Sahte örnek metadata (sample_id.json)
└── splits/
    ├── train.json
    ├── val.json
    └── test.json
```

- **Konuşmacı ayrımlı split:** Aynı `speaker_id` yalnızca bir split’te (train / val / test) bulunur.

---

## 3. Örnek Metadata Şeması

Her örnek için bir JSON (metadata veya fakes/metadata içinde):

| Alan | Açıklama |
|------|----------|
| `sample_id` | Benzersiz ID (örn. TRDF_00021) |
| `speaker_id` | Konuşmacı ID (örn. spk_03) |
| `video_path` | Proje köküne göre video yolu |
| `audio_path` | Proje köküne göre ses yolu |
| `transcript_tr` | Türkçe transkript |
| `label_main` | Ana sınıf (aşağıda) |
| `label_visual_fake` | 0/1 görsel sahte mi |
| `label_audio_fake` | 0/1 ses sahte mi |
| `label_sync` | 1=uyumlu, 0=uyumsuz |
| `sync_shift_ms` | Varsa temporal kayma (ms) |
| `mismatch_type` | temporal_shift, content_mismatch, synthetic_audio, none vb. |
| `domain` | academic vb. |
| `duration_sec` | Süre (saniye) |
| (opsiyonel) | `faces_dir`, `mouths_dir`, `source_sample_id` vb. |

---

## 4. Sınıflar (label_main)

| Sınıf | Açıklama |
|--------|----------|
| `real_sync` | Gerçek video, gerçek ses, doğru transcript, senkron doğru. Pozitif referans. |
| `fake_sync_shift` | Video gerçek; ses aynı içerik ama zamanda kaydırılmış (örn. +200 ms, +400 ms). |
| `fake_content_mismatch` | Video gerçek; başka cümle veya başka kişi sesi eşlenmiş. |
| `fake_audio_synthetic` | Video gerçek; ses sentetik/TTS/klon. |
| `fake_visual` | (Opsiyonel) Yüz manipülasyonu; ses doğal olabilir. |

---

## 5. Sahte Örnek Üretim Yöntemleri

- **Temporal shift:** Aynı WAV’ı belirli ms kaydırıp yeni ses dosyası yazma; `sync_shift_ms` ∈ {100, 200, 300, 400, 500}.  
- **Content mismatch:** Aynı veya farklı konuşmacıya ait başka bir örneğin sesini bu videoya atama; metadata’da farklı `transcript_tr` ve `audio_path`.  
- **Synthetic audio:** TTS/klon çıktısı (veya stub: aynı sürede sessiz dosya); `label_audio_fake=1`.  

Üretim script’i: `src/preprocessing/generate_fakes.py`; pipeline: `python run_pipeline.py generate_fakes`.

---

## 6. Veri Toplama Protokolü (Hedef)

- **Konuşmacı:** 20–40 kişi (v1: 20).  
- **Video:** Kişi başı 10–20 kısa video, video başı 4–8 saniye.  
- **Kurallar:** Yüz net, ağız kapanmıyor, tek kişi, Türkçe konuşma, ses temiz, kamera sabit/yarı sabit.  
- **Cümle havuzu:** Fonetik çeşitlilik (b, p, m, f, v, o, u, a, e vb.) içeren Türkçe cümleler.

---

## 7. Pipeline Sırası

1. **Ham videoları** `data/raw_videos/` içine koy (konuşmacı/video isimlendirmesi tutarlı olsun).  
2. **Ön işleme:**  
   `python run_pipeline.py preprocess --video-list video_list.csv`  
   (video_list: `video_path,sample_id,speaker_id` her satırda.)  
3. **Sahte üretimi:**  
   `python run_pipeline.py generate_fakes`  
4. **Split:**  
   `python run_pipeline.py build_splits`  
5. **Eğitim:**  
   `python run_pipeline.py train --model visual`  
   `python run_pipeline.py train --model sync --config configs/train_sync.yaml`  
   `python run_pipeline.py train --model fusion --config configs/fusion.yaml`  
6. **Değerlendirme:**  
   `python run_pipeline.py evaluate` (ve `src/evaluation/` script’leri).

---

## 8. Hızlı Doğrulama

- Her örnek için `transcript_tr` ve uygun `label_sync` var mı?  
- `sync_shift_ms` sadece `fake_sync_shift` örneklerinde dolu mu?  
- Aynı `speaker_id` hem train hem test’te görünmüyor mu?

Bu kontroller, deneyin tekrarlanabilir ve makale iddialarına uygun olması için kritiktir.
