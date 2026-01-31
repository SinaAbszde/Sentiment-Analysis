# 🧠 Smart Bilingual Sentiment Analyzer

[![.NET 8](https://img.shields.io/badge/.NET-8.0-blue.svg)](https://dotnet.microsoft.com/download/dotnet/8.0)
[![ML.NET](https://img.shields.io/badge/Machine%20Learning-ML.NET-orange.svg)](https://dotnet.microsoft.com/apps/machinelearning-ai/ml-dotnet)
[![Status](https://img.shields.io/badge/Status-Completed-green.svg)]()

یک ابزار هوشمند برای تحلیل احساسات متون فارسی و انگلیسی که با استفاده از **ML.NET** و معماری **Mashup** طراحی شده است.

---

## 🌟 قابلیت‌های کلیدی (Features)

- **پشتیبانی دو زبانه (Bilingual):** تشخیص هوشمند زبان و تحلیل حس متن (فارسی و انگلیسی).
- **دیتای عظیم (Big Data):** آموزش دیده روی بیش از **۱۱۹,۰۰۰** رکورد واقعی (ترکیبی از IMDB، دیجی‌کالا و اسنپ‌فود).
- **قابلیت Mashup:** اتصال زنده به APIهای خارجی (Advice Slip & Quotable) برای تست تصادفی مدل.
- **منطقه خنثی (Neutral Zone):** تشخیص متون خبری و بی‌طرف برای جلوگیری از تحلیل‌های غلط.
- **رابط کاربری مدرن:** طراحی RTL و کاملاً ریسپانسیو با فونت وزیر.

---

## 🏗 معماری پروژه (Architecture)

این پروژه از ترکیب دو بخش اصلی ساخته شده است:
1. **Model Trainer (ML.NET):** پیش‌پردازش داده‌ها و آموزش مدل با استفاده از الگوریتم‌های طبقه‌بندی.
2. **Web API (.NET 8):** سرویس‌دهی پیش‌بینی‌ها و مدیریت رابط کاربری.



---

## 🚀 شروع سریع (Quick Start)

### پیش‌نیازها
- [.NET 8.0 SDK](https://dotnet.microsoft.com/download/dotnet/8.0)
- Rider یا Visual Studio 2022

### نصب و اجرا
۱. مخزن را کلون کنید:
   ```bash
   git clone https://github.com/SinaAbszde/Sentiment-Analysis.git
   ```
۲. فایل **SentimentModel.zip** را از بخش [Releases](https://github.com/SinaAbszde/Sentiment-Analysis/releases/tag/v1.0.0) دانلود کنید.
۳. فایل مدل را در پوشه `SentimentAnalysis.API/MLModels` قرار دهید.
۴. پروژه را اجرا کنید:
```bash
dotnet run
```
## 🛠 تکنولوژی‌های مورد استفاده
- **Backend:** C# (.NET 8 Web API)
- **Machine Learning:** ML.NET
- **Frontend:** HTML5, CSS3, JavaScript (Vanilla)
---

## 👨‍💻 نویسنده
**Sina Abaszadeh** - .NET Web API Developer
