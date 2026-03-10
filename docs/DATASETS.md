# Hazır Veri Setleri — Deepfake TR

Projeyi kendi videolarını toplamadan çalıştırmak için kullanılabilecek hazır veri setleri ve indirme linkleri.

---

## 1. AVLips (Lip-sync Deepfake) — Önerilen

**Ne işe yarar:** Dudak–ses uyumsuzluğu (S_l) ve lip-sync deepfake tespiti için. Real + fake lip-sync videoları içerir.

| Bilgi | Değer |
|-------|--------|
| **Kaynak** | LipFD – "Lips Are Lying" (NeurIPS 2024) |
| **Repo** | https://github.com/AaronComo/LipFD |
| **İndirme** | [Google Drive – AVLips v1.0](https://drive.google.com/file/d/1fEiUo22GBSnWD7nfEwDW86Eiza-pOEJm/view?usp=share_link) |
| **Boyut** | Sıkıştırılmış ~9 GB, açılmış ~16 GB |
| **Dil** | İngilizce (Türkçe değil; yine de lip-sync ve görsel model için kullanılabilir) |

**Projeye aktarma:**  
`python scripts/download_dataset.py avlips` (gdown ile indirir, `data/raw_videos/avlips/` altına kopyalar).

---

## 2. FakeAVCeleb (Görsel + Ses Multimodal)

**Ne işe yarar:** Hem görsel hem ses deepfake; lip-sync ve sahte ses örnekleri. S_v ve S_l için uygun.

| Bilgi | Değer |
|-------|--------|
| **Kaynak** | DASH-Lab FakeAVCeleb |
| **Site** | https://sites.google.com/view/fakeavcelebdash-lab/ |
| **İndirme** | Form doldurmanız gerekir: [İndirme formu](https://bit.ly/38prlVO) veya [Google Form](https://docs.google.com/forms/d/e/1FAIpQLSfPDd3oV0auqmmWEgCSaTEQ6CGpFeB-ozQJ35x-B_0Xjd93bw/viewform) |
| **İçerik** | 20.000+ audio-video deepfake, 5 etnik grup, hem video hem sentetik ses |

Form onaylandıktan sonra indirme scripti e‑posta ile gelir.

---

## 3. Hugging Face — Hızlı deneme

**AV-Deepfake1M (ControlNet)**  
- Dataset: https://huggingface.co/datasets/ControlNet/AV-Deepfake1M  
- Örnek indirme: `huggingface-cli download ControlNet/AV-Deepfake1M --repo-type dataset --local-dir ./data/external/AV-Deepfake1M`

**UniDataPro/deepfake-videos-dataset**  
- 10.000+ video, 7.000+ kişi, AI yüz overlay  
- https://huggingface.co/datasets/UniDataPro/deepfake-videos-dataset  

**Not:** Bu setlerin etiket şeması bizimkinden farklı; indirdikten sonra `scripts/convert_foreign_dataset.py` ile kendi metadata formatımıza dönüştürbilirsin.

---

## 4. VoxCeleb2 (Gerçek konuşan yüz – sadece real)

**Ne işe yarar:** Sadece gerçek videolar. Üzerine kendin fake_sync_shift / content_mismatch üreterek proje formatına getirebilirsin.

| Bilgi | Değer |
|-------|--------|
| **Site** | https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html |
| **İndirme** | Kayıt formu: [VoxCeleb form](https://docs.google.com/forms/d/e/1FAIpQLSdQhpq2Be2CktaPhuadUMU7ZDJoQuRlFlzNO45xO-drWQ0AXA/viewform) |
| **Script** | https://github.com/walkoncross/voxceleb2-download |
| **Boyut** | Video >300 GB, ses >72 GB |
| **Dil** | Çok dilli (Türkçe az; çoğunlukla İngilizce) |

---

## 5. Türkçe odaklı

**504 Saat Türkçe konuşma (Nexdata)**  
- Sadece ses; video yok.  
- https://github.com/Nexdata-AI/504-Hours-Turkish-Real-world-Casual-Conversation-and-Monologue-speech-dataset  

Şu an için **doğrudan Türkçe konuşan yüz + deepfake** hazır veri seti yok; ya VoxCeleb/FakeAVCeleb ile İngilizce çalışırsın ya da az sayıda Türkçe videoyu kendin toplayıp AVLips/FakeAVCeleb tarzı etiketleri taklit edebilirsin.

---

## Hızlı başlangıç (AVLips ile)

```bash
pip install gdown
python scripts/add_real_videos.py avlips --download
# Zip acildiktan sonra: python scripts/add_real_videos.py avlips
python run_pipeline.py generate_fakes
python run_pipeline.py build_splits
```

Bu adımlardan sonra `train` ve `evaluate` pipeline’ı kendi split’lerinle kullanılabilir.
