## bitirme tezimde adım adım yapılacaklar 

ADIM 1 — Veri Toplama

Günlük baraj doluluk oranlarını topla.

Meteorolojik verileri topla:

Yağış

Sıcaklık

Buharlaşma

Tarih bilgisini datetime formatına çevir.

ADIM 2 — Veri Ön İşleme

Tarihe göre sırala.

Eksik değerleri:

Küçük boşluklar → Interpolation

Büyük boşluklar → Ortalama ile doldurma

Aykırı değer kontrolü yap.

ADIM 3 — Feature Engineering

Lag özellikleri oluştur:

t-1

t-7

t-30

7 günlük moving average oluştur.

Ay ve mevsim bilgisi ekle.

Sliding window oluştur:

Geçmiş 7 gün → 8. gün tahmini

ADIM 4 — Train-Test Bölme

⚠ Shuffle yapılmaz.

İlk %80 → Train

Son %20 → Test

ADIM 5 — Normalizasyon

MinMaxScaler kullan.

Scaler sadece train verisi ile fit edilir.

Test verisi transform edilir.

ADIM 6 — Baseline Model

Naive Model (Yarın = Bugün)

Linear Regression

Performans karşılaştırması yapılır.

ADIM 7 — LSTM Model Tasarımı

Model yapısı:

Input → LSTM(64) → Dropout(0.2) → Dense(32) → Output(1)

ADIM 8 — Model Derleme

Optimizer: Adam

Loss: MSE

Metric: MAE

ADIM 9 — Model Eğitimi

Epoch: 100

Batch size: 32

EarlyStopping kullan

ADIM 10 — Tahmin ve Performans

Test seti üzerinde tahmin yap.

Ölçek geri dönüşümü yap.

RMSE, MAE, R² hesapla.

Gerçek vs Tahmin grafiği çiz.

Sonuçlar anlamlı mı, değilse ADIM 2 ye git 

### Açıklama
- **Daire (Başlangıç/Bitiş):** sürecin giriş ve çıkış noktaları  
- **Dikdörtgen:** işlem adımları (örneğin “Veri Toplama”)  
- **Elmas (Decision):** karar noktası (“Sonuçlar anlamlı mı?”)  
- **Oklar:** akış yönü
- 
```mermaid
flowchart TD
    A([Başlangıç]) --> B[Adım 1: Veri Toplama]
    B --> C[Adım 2: Veri Ön İşleme]
    C --> D[Adım 3: Feature Engineering]
    D --> E[Adım 4: Train-Test Bölme]
    E --> F[Adım 5: Normalizasyon]
    F --> G[Adım 6: Baseline Model]
    G --> H[Adım 7: LSTM Model Tasarımı]
    H --> I[Adım 8: Model Derleme]
    I --> J[Adım 9: Model Eğitimi]
    J --> K[Adım 10: Tahmin ve Performans]
    K --> L{Sonuçlar anlamlı mı?}
    L -->|Evet| M([Bitiş])
    L -->|Hayır| C
