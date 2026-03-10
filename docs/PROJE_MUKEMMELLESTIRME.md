# Projeyi Mükemmelleştirme — Makale Sonucu İçin

Makale sonucunun ikna edici olması için **proje** (veri, model, değerlendirme) sağlam olmalı. Aşağıdakiler öncelik sırasıyla.

---

## 1. VERİ (En büyük etki)

| Öneri | Ne yapılır | Neden |
|-------|-------------|--------|
| **Gerçek / daha büyük veri** | `python scripts/add_real_videos.py avlips --download` veya URL listesi ile Türkçe videolar ekle; hedef 200+ real, 500+ fake. | Demo/sentetik ile AUC ~0.5’e takılır; gerçek veri skorları anlamlı yapar. |
| **Denge** | Real/fake oranı train’de en az 1:3–1:5; çok dengesizse oversampling veya loss ağırlığı (class weight). | Hep “fake” demesi precision/accuracy’yi düşürür. |
| **Speaker sayısı** | 20+ konuşmacı, speaker-disjoint split (zaten var); mümkünse 50+ speaker. | Genelleme; “kişi ezberlemedi” iddiası. |
| **Manipülasyon çeşitliliği** | real_sync, sync_shift, content_mismatch, synthetic_audio (zaten var); her türden yeterli örnek. | Ablasyon: “hangi manipülasyonda fusion iyi?” diyebilirsin. |

**Pratik adım:** Önce AVLips veya 20–30 gerçek video (URL listesi) ekle → `generate_fakes` + `build_splits` → yeniden eğit.

---

## 2. MODEL EĞİTİMİ

| Öneri | Ne yapılır | Neden |
|-------|-------------|--------|
| **Tam epoch** | Demo yerine tam config: `train_visual.yaml` (30 epoch), `train_sync.yaml` (40 epoch). `run_pipeline.py full --demo --large --full-config`. | 2 epoch ile Visual AUC 0.5 kalır; daha fazla epoch öğrenme şansı. |
| **Learning rate / early stop** | Val loss artınca dur veya lr decay; overfitting varsa augmentation (yüz: hafif crop, flip). | Overfitting küçük veride hızlı; val metrikleri izle. |
| **Fusion α** | Şu an sabit 0.5; ablasyonda α=0.25 iyi çıktıysa makalede “önerilen α=0.25” de. İleride α’yı öğrenilebilir yap. | Makale: “Senkron ağırlığı artınca doğruluk arttı” net yazılır. |
| **Görsel backbone** | ResNet18 yeterli; veri büyürse EfficientNet veya küçük bir ViT deneyebilirsin. | Önce veri ve epoch; sonra mimari. |

**Pratik adım:** Mevcut 818 örnek ile `--full-config` ile visual + sync + fusion eğit; sonuçları karşılaştır.

---

## 3. DEĞERLENDİRME VE RAPORLAMA

| Öneri | Ne yapılır | Neden |
|-------|-------------|--------|
| **Optimal eşik** | Zaten Youden (ROC) kullanılıyor; sabit 0.5 kullanma. | Skor dağılımı 0.5’te değilse accuracy/recall anlamsız olur. |
| **Çoklu çalıştırma** | 3–5 farklı seed ile train; Acc ve AUC için ortalama ± std raporla. | “Tek deneme” eleştirisi; güven aralığı makaleyi güçlendirir. |
| **Manipülasyon tipine göre** | Test setinde fake_sync_shift, content_mismatch, synthetic_audio ayrı ayrı Acc/AUC (veya en azından Fusion). | “Fusion özellikle sync_shift’te iyi” gibi net cümle. |
| **ROC + confusion matrix** | `roc_curve.png`, `confusion_matrix.png` üreten script (evaluate sonrası); makalede Şekil olarak koy. | Görsel kanıt; reviewer ister. |

**Pratik adım:** `scripts/export_results_latex.py` ve `run_ablation_alpha.py` ile tabloları güncelle; isteğe bağlı ROC/confusion script ekle.

---

## 4. TEKRARLANABILIRLIK VE DOKÜMANTASYON

| Öneri | Ne yapılır | Neden |
|-------|-------------|--------|
| **Sabit seed** | `train_*.py` ve `build_splits` içinde `random.seed(42)`, `torch.manual_seed(42)` kullan. | Aynı komutla aynı sonuç. |
| **Config tek yerde** | Tüm hyperparameter’lar `configs/*.yaml`; kodda sabit değer azalt. | “Deneyler şu config ile” tek cümle. |
| **requirements.txt** | Versiyon sabitle (torch, sklearn, opencv). | “pip install -r requirements.txt” ile aynı ortam. |
| **README** | “Tam pipeline: full --demo --large --full-config”; “Tabloları güncelle: export_results_latex.py + run_ablation_alpha.py”. | Makale sonucu = proje çıktısı; okuyan tekrarlasın. |

---

## 5. MAKALE SONUCUNA DOĞRUDAN YANSıYACAKLAR

- **“Sınırlı veri”** yazacaksan → Veriyi büyüt (AVLips / URL); sonra “artırılmış veri ile Acc/AUC şu seviyeye çıktı” diyebilirsin.
- **“Multimodal fusion tek modaldan iyi”** diyebilmek için → Fusion AUC > Visual AUC ve (tercihen) Sync AUC; Youden eşik ile Acc da makul olsun.
- **“α=0.25 daha iyi”** diyebilmek için → Ablasyon tablosunda α=0.25 satırı gerçekten daha iyi olmalı (zaten öyle çıkıyor).
- **“Türkçe için uygun”** diyebilmek için → En azından Türkçe cümle havuzu + protokol var; mümkünse az sayıda gerçek Türkçe video ile doğrula.

---

## Kısa özet: Şu an ne yapılsın?

1. **Veri:** AVLips veya URL listesi ile gerçek videolar ekle; 200+ örnek hedefle.
2. **Eğitim:** Tam config (30/40 epoch) ile visual + sync + fusion eğit.
3. **Rapor:** Youden eşik (zaten var), isteğe bağlı 3 seed ortalama ± std, ROC/confusion figürü.
4. **Makale:** Sonuçları projeden üret (tablolar \input ile); “sınırlı veri / gelecek çalışma” net yaz.

Proje bu hale gelince makale sonucu “önemli bir sinyal”, “multimodal fusion katkısı” ve “gelecek çalışmalar” ile tutarlı ve ikna edici olur.
