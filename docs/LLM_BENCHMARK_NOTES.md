# LLM Benchmark — Yorum Notları

Bu dosya, `paper/llm_benchmark.md` sonuçlarını makale yazarken hatırlamak için kısa yorumlar içerir.

## 1. Mevcut sonuçlar (AVLips, test split, n=800)

Tablodaki değerler:

| Model         | Accuracy | AUC   | EER   |
|--------------|----------|-------|-------|
| simple       | 0.351    | 0.500 | 1.000 |
| whisper_sbert| 0.351    | 0.499 | 1.000 |

Özet:

- **Basit text encoder** ile **Whisper+SBERT** arasında anlamlı fark yok.
- Her iki durumda da AUC ≈ 0.5 ve EER = 1.0 → metin–dudak hizalaması şu aşamada rastgele seviyede.

## 2. Çıkarımlar 

- *\"LLM tabanlı metin embedding'lerini (Whisper+SBERT) basit bir text encoder ile karşılaştırdık (Tablo X). Mevcut dudak embedding'leriyle birlikte kullanıldığında her iki yöntem de AUC ≈ 0.5 civarında performans göstermiştir. Bu durum, darboğazın metin tarafındaki dil modeli değil, dudak hareketinden elde edilen temsil ve genel multimodal hizalama olduğunu göstermektedir.\"*

- *\"Sonuçlar, metin tarafında güçlü LLM embedding'leri kullanmanın tek başına yeterli olmadığını, dudak embedding'i için daha güçlü önceden eğitilmiş AV-sync modellerine (örneğin SyncNet) ihtiyaç olduğunu ortaya koymaktadır. Gelecek çalışmalarda bu tür backbone'lar ve multimodal transformer mimarileri incelenecektir.\"*

- *\"Whisper+SBERT konfigürasyonunun simple encoder'a göre belirgin bir kazanç sağlamaması, şu aşamada sistemin LLM kapasitesinden tam olarak faydalanamadığını göstermektedir. Yine de bu ön deney, çerçevenin farklı LLM tabanlı embedding'leri sistematik olarak karşılaştırmak için uygun bir test yatağı sunduğunu göstermektedir.\"*

## 3. Makalede nereye koyulur?

- **\"LLM Benchmark Analizi\"** alt başlığının altında:
  - Tablo: `\input{llm_benchmark}`.
  - Yukarıdaki 2–3 cümle, ufak düzenlemelerle doğrudan kullanılabilir.

