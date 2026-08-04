# کد تولید نمودارهای پایان‌نامه

## ساختار
- `parse_results.py` — تمام فایل‌های JSON پوشهٔ `results/` (که خودتان آپلود کرده بودید) را می‌خواند و دو دیتافریم پانداس (`eval_df`, `train_df`) می‌سازد و به‌صورت pickle (`eval_df.pkl`, `train_df.pkl`) و CSV ذخیره می‌کند. **باید اول از همه اجرا شود.**
- `plot_style.py` — تنظیمات مشترک matplotlib (رنگ‌ها، فونت، برچسب‌های شرایط) که همهٔ اسکریپت‌های `figures/` از آن import می‌کنند.
- `figures/fig01_baseline.py` … `figures/fig08_perplexity.py` — هر کدام یکی از نمودارهای فصل ۴ را تولید می‌کنند (خروجی: PDF).
- `architecture_diagram/architecture_diagram.tex` — کد TikZ نمودار معماری فصل ۳ (مستقل از پایتون؛ با `xelatex` کامپایل می‌شود).

## نحوهٔ اجرا
```bash
pip install pandas matplotlib numpy

# ۱. مسیر RESULTS_DIR را در بالای parse_results.py به مسیر واقعی پوشهٔ results خودتان تغییر دهید
python3 parse_results.py

# ۲. اسکریپت‌های هر نمودار را از همان پوشه اجرا کنید (به eval_df.pkl/train_df.pkl نیاز دارند)
cd figures
python3 fig01_baseline.py
python3 fig02_gain_vs_lambda.py
python3 fig03_layers.py        # دو شکل تولید می‌کند: gain_by_n_layers, gain_by_target_layer
python3 fig04_profile.py       # نمایهٔ عمق: خط پایه در برابر پیکربندیِ توصیه‌شده (لایه ۵، C4، λ=۰.۳)
python3 fig05_training.py      # پویایی آموزش، ۴ پانل
python3 fig06_fidelity_heatmap.py
python3 fig07_interpretability.py
python3 fig08_perplexity.py

# ۳. نمودار معماری (جداگانه، با xelatex)
cd ../architecture_diagram
xelatex architecture_diagram.tex
```

هر اسکریپت خروجی PDF را در مسیر `OUT_DIR` (تعریف‌شده در `plot_style.py`) می‌نویسد؛ پیش از اجرا این مسیر را هم به پوشهٔ دلخواه خودتان تغییر دهید.

## نکات
- `fig04_profile.py` و `fig07_interpretability.py` به‌صورت پیش‌فرض پیکربندیِ توصیه‌شده (لایهٔ ۵، شرط C4، λ=۰.۳) را رسم می‌کنند، نه پیکربندیِ با بیشترین بهبودِ خام (λ=۱.۰)؛ اگر خواستید نمودار را برای پیکربندیِ دیگری بسازید، مقادیرِ `layer_setting`, `condition`, `lam` را در همان اسکریپت عوض کنید.
- تمام اعداد و برچسب‌های نمودارها به انگلیسی نوشته شده‌اند (رویهٔ رایج در نمودارهای فنی حتی داخل پایان‌نامه‌های فارسی)، و کپشنِ فارسی هر شکل در خودِ فایل لتک (`chapter3.tex`/`chapter4.tex`) آمده است.
