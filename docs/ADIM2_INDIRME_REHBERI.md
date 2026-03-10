# Adım 2: Gerçek videoları nereden, nasıl indirirsin?

İki yol var. **Birini seç.**

---

## Yol A — AVLips (hazır veri seti, tek link)

**Nereden:** Google Drive (LipFD / “Lips Are Lying” veri seti).  
**Boyut:** Sıkıştırılmış ~9 GB, açılmış ~16 GB.  
**Dil:** İngilizce (Türkçe değil; lip-sync ve görsel model için yine de kullanılır).

### Nasıl indirirsin?

**1) Gerekli paket**

```bash
pip install gdown
```

**2) Proje klasöründen indir (otomatik)**

```bash
cd deepfake_tr_project
python scripts/add_real_videos.py avlips --download
```

Bu komut indirmeyi başlatır; dosya `data/raw_videos/avlips/AVLips.zip` olarak iner (~9 GB, süre internete göre değişir).  
**Komut çalışmazsa:** Tarayıcıdan manuel indir: https://drive.google.com/file/d/1fEiUo22GBSnWD7nfEwDW86Eiza-pOEJm/view — indirdiğin zip’i `data/raw_videos/avlips/AVLips.zip` konumuna koy, sonra zip’i açıp **4)** adımına geç.

**3) Zip’i aç**

- **Windows:** `data/raw_videos/avlips` klasörüne git → `AVLips.zip` üzerine sağ tık → “Tümünü çıkart” (veya 7-Zip / WinRAR).
- **Hedef:** Aynı klasörde `AVLips` adında bir klasör çıksın; içinde `real` ve `fake` alt klasörleri olsun.

**4) Dönüştür + yüz/ağız çıkar**

Zip açıldıktan sonra:

```bash
python scripts/add_real_videos.py avlips
```

Bu adım: AVLips’i proje formatına çevirir ve her video için yüz/agiz görsellerini üretir. Bittikten sonra Adım 3’e geçebilirsin (`generate_fakes`).

---

## Yol B — Kendi linklerin (YouTube, TED vb.)

**Nereden:** Sen belirliyorsun (Türkçe konuşma videoları önerilir).  
**Boyut:** İndirdiğin video sayısına bağlı.

### Nasıl yaparsın?

**1) CSV dosyası oluştur**

Projede `data/video_urls.csv` adında bir dosya oluştur. İçeriği örnek:

```csv
url,sample_id,speaker_id
https://www.youtube.com/watch?v=VIDEO_ID_1,tr_001,spk_1
https://www.youtube.com/watch?v=VIDEO_ID_2,tr_002,spk_1
https://www.ted.com/talks/...,tr_003,spk_2
```

- **url:** Video linki (YouTube, TED, vb.).
- **sample_id:** Örnek adı (benzersiz; örn. tr_001, tr_002).
- **speaker_id:** Konuşmacı adı (aynı kişi = aynı speaker_id).

En az 10–20 video ile başlamak mantıklı. Örnek şablon: `data/video_urls_example.csv` dosyasını `data/video_urls.csv` olarak kopyalayıp `EXAMPLE1`, `EXAMPLE2` yerine gerçek video ID’lerini yazabilirsin.

**2) yt-dlp kur**

```bash
pip install yt-dlp
```

**3) İndir + ön işle**

```bash
cd deepfake_tr_project
python scripts/add_real_videos.py urls --csv data/video_urls.csv
```

Bu komut: CSV’deki her URL’yi indirir, yüz/agiz çıkarır ve metadata üretir. Bittikten sonra Adım 3’e geç (`generate_fakes`).

---

## Özet

| Yol | Komut | Not |
|-----|--------|-----|
| **A – AVLips** | `add_real_videos.py avlips --download` → zip aç → `add_real_videos.py avlips` | ~9 GB indirme, İngilizce |
| **B – URL listesi** | CSV hazırla → `add_real_videos.py urls --csv data/video_urls.csv` | Türkçe videolar ekleyebilirsin |

İndirme ve işlem bittikten sonra sıradaki adım: **Adım 3** — `python run_pipeline.py generate_fakes`
