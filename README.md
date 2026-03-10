# Makale — Sonuçlar projeden üretilir

**Makale metni örnek şablondur.** Tablo 2 (test sonuçları) ve Tablo 3 (fusion α ablasyonu) sayıları proje çalıştırıldığında üretilen sonuçlarla doldurulur; sabit değerler değildir.

## Tabloları güncellemek

Projede eğitim/değerlendirme yaptıktan sonra makaledeki tabloları güncellemek için:

```bash
# Proje kökünden (deepfake_tr_project)
python scripts/export_results_latex.py --split test --out paper/results_table.tex
python scripts/run_ablation_alpha.py --split test --out paper/ablation_alpha.tex
```

- **Tablo 2 (results_table.tex):** Visual, Sync, Fusion — Acc, Prec, Rec, F1, AUC, EER (Youden eşik ile).
- **Tablo 3 (ablation_alpha.tex):** α = 0.25, 0.5, 0.75 için Fusion sonuçları.

LaTeX tarafında bu dosyalar `\input{results_table}` ve `\input{ablation_alpha}` ile eklenebilir; böylece makale derlendiğinde her zaman **projenin ürettiği sonuçlar** kullanılır.

## Özet

| Ne | Nereden gelir |
|----|-----------------|
| Tablo 2 (test sonuçları) | `export_results_latex.py` |
| Tablo 3 (α ablasyon) | `run_ablation_alpha.py` |
| Makale metni | Örnek / şablon; sonuçları sen belirlemiyorsun, proje belirliyor. |
