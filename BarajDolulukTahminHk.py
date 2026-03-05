# -*- coding: utf-8 -*-   # Dosya Türkçe karakterleri desteklesin diye kodlama ayarı
# İzmir Baraj Doluluk Oranı Tahmini - LSTM Modeli
# Şimdilik Seattle verisiyle çalışıyoruz (sentetik reservoir_percent ile)

import numpy as np               # Sayısal hesaplamalar için kullanılan kütüphane
import pandas as pd              # Veri okuma ve işleme için kullanılan kütüphane
import matplotlib.pyplot as plt  # Grafik çizmek için kullanılan kütüphane
from sklearn.preprocessing import MinMaxScaler   # Verileri belli aralığa ölçeklemek için
from sklearn.linear_model import LinearRegression # Basit doğrusal regresyon modeli
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score # Model performans ölçütleri
from tensorflow.keras.models import Sequential   # Keras ile model kurmak için
from tensorflow.keras.layers import LSTM, Dense, Dropout # LSTM ve ek katmanlar
from tensorflow.keras.callbacks import EarlyStopping # Eğitimde erken durdurma için

# ============================================================
# ADIM 1 - Veri Yükleme
# ============================================================
data = pd.read_csv(r"C:\Users\acer\Desktop\310.py\tezzz.py\seattle-weather.csv") # CSV dosyasını oku
data['date'] = pd.to_datetime(data['date']) # Tarih sütununu tarih tipine çevir
data = data.sort_values('date').reset_index(drop=True) # Tarihe göre sırala ve indexleri sıfırla
print("Veri yüklendi. Satır sayısı:", len(data)) # Kaç satır veri olduğunu yazdır

# Sentetik baraj doluluk oranı (gerçek veri yok, hava durumundan türetiyoruz)
data['reservoir_percent'] = (
    50
    + 0.3 * data['precipitation'] # Yağış arttıkça doluluk artar
    - 0.1 * data['temp_max']      # Sıcaklık arttıkça buharlaşma → doluluk azalır
    + 0.05 * data['wind']         # Rüzgarın küçük etkisi
).clip(0, 100) # Değerleri 0-100 arası sınırla

# ============================================================
# ADIM 2 - Eksik Veri Doldurma
# ============================================================
data['month'] = data['date'].dt.month # Ay bilgisini çıkar
for col in ['reservoir_percent', 'precipitation', 'temp_max', 'temp_min', 'wind']:
    data[col] = data[col].interpolate(method='linear') # Eksikleri doğrusal tahminle doldur
    data[col] = data.groupby('month')[col].transform(lambda x: x.fillna(x.mean())) # Aynı ayın ortalamasıyla doldur

# Aykırılık raporu (uç değerleri sayıyoruz)
print("\nAykırılık raporu:")
for col in ['precipitation', 'temp_max', 'temp_min', 'wind', 'reservoir_percent']:
    Q1 = data[col].quantile(0.25) # 1. çeyrek
    Q3 = data[col].quantile(0.75) # 3. çeyrek
    IQR = Q3 - Q1                 # Çeyrekler arası fark
    n = ((data[col] < Q1 - 1.5 * IQR) | (data[col] > Q3 + 1.5 * IQR)).sum() # Uç değer sayısı
    print(f"  {col}: {n} aykırı değer")

# ============================================================
# ADIM 3 - Özellik Çıkarma (Feature Engineering)
# ============================================================
target = 'reservoir_percent' # Tahmin etmek istediğimiz değişken

# Gecikmeli değerler (lag) ve hareketli ortalama ekliyoruz
data['lag_1']    = data[target].shift(1)   # 1 gün önceki değer
data['lag_7']    = data[target].shift(7)   # 1 hafta önceki değer
data['lag_30']   = data[target].shift(30)  # 1 ay önceki değer
data['rolling_7']= data[target].rolling(window=7).mean() # 7 günlük ortalama
data['season']   = data['month'].map({     # Ay → Mevsim
    12:1, 1:1, 2:1,  # Kış
    3:2,  4:2, 5:2,  # İlkbahar
    6:3,  7:3, 8:3,  # Yaz
    9:4, 10:4, 11:4  # Sonbahar
})

data = data.dropna().reset_index(drop=True) # Eksik satırları at

# Kullanılacak özellikler
features = ['precipitation', 'temp_max', 'temp_min', 'wind',
            'lag_1', 'lag_7', 'lag_30', 'rolling_7', 'month', 'season']

X_raw = data[features].values # Girdi verileri
y_raw = data[target].values   # Çıktı (doluluk oranı)
print("\nÖzellik sayısı:", len(features))
print("Toplam örnek sayısı:", len(X_raw))

# ============================================================
# ADIM 4 - Train-Test Bölme
# ============================================================
split = int(len(X_raw) * 0.8) # %80 eğitim, %20 test
X_train_raw, X_test_raw = X_raw[:split], X_raw[split:] # Eğitim ve test girdileri
y_train_raw, y_test_raw = y_raw[:split], y_raw[split:] # Eğitim ve test çıktıları
print(f"\nTrain: {len(X_train_raw)} satır | Test: {len(X_test_raw)} satır")

# ============================================================
# ADIM 5 - Normalizasyon
# ============================================================
scaler_X = MinMaxScaler() # Girdiler için ölçekleyici
scaler_y = MinMaxScaler() # Çıktılar için ölçekleyici

X_train_scaled = scaler_X.fit_transform(X_train_raw) # Eğitim verisine göre ölçekle
X_test_scaled  = scaler_X.transform(X_test_raw)      # Test verisini aynı ölçekle
y_train_scaled = scaler_y.fit_transform(y_train_raw.reshape(-1, 1)) # Çıktıları ölçekle
y_test_scaled  = scaler_y.transform(y_test_raw.reshape(-1, 1))

# ============================================================
# ADIM 6 - Baseline Modeller
# ============================================================
# Naive model: yarın = bugün
y_naive        = y_test_raw[:-1]       # Bugünkü değer
y_naive_actual = y_test_raw[1:]        # Yarınki gerçek değer
rmse_naive = np.sqrt(mean_squared_error(y_naive_actual, y_naive)) # Hata ölçümü
mae_naive  = mean_absolute_error(y_naive_actual, y_naive)         # Ortalama mutlak hata
r2_naive   = r2_score(y_naive_actual, y_naive)                    # R2 skoru

# Doğrusal Regresyon
lr = LinearRegression()
lr.fit(X_train_scaled, y_train_raw) # Modeli eğit
y_lr_pred = lr.predict(X_test_scaled) # Test verisiyle tahmin yap
rmse_lr = np.sqrt(mean_squared_error(y_test_raw, y_lr_pred)) # Hata ölçümü
mae_lr  = mean_absolute_error(y_test_raw, y_lr_pred)         # Ortalama mutlak hata
r2_lr   = r2_score(y_test_raw, y_lr_pred)                    # R2 skoru

# ============================================================
# ADIM 7 - Sliding Window
# ============================================================
WINDOW = 30 # 30 günlük geçmiş → 31. günü tahmin

def create_sequences(X, y, window):
    Xs, ys = [], []
    for i in range(len(X) - window):
        Xs.append(X[i:i + window]) # 30 günlük veri
        ys.append(y[i + window])   # 31. günün çıktısı
    return np.array(Xs), np.array(ys)

X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, WINDOW) # Eğitim dizileri
X_test_seq,  y_test_seq  = create_sequences(X_test_scaled,  y_test_scaled,  WINDOW) # Test dizileri
print(f"\nSequence shape: {X_train_seq.shape}")

# ============================================================
# ADIM 8 - LSTM Model Tasarımı
# ============================================================
model = Sequential([ # Modeli sırayla katmanlar ekleyerek kuruyoruz
    LSTM(64, return_sequences=False, input_shape=(WINDOW, len(features))), # 64 nöronlu LSTM katmanı, giriş boyutu: 30 gün x özellik sayısı
    Dropout(0.2),   # %20 oranında dropout, aşırı öğrenmeyi (overfitting) engeller
    Dense(32, activation='relu'), # 32 nöronlu tam bağlı katman, ReLU aktivasyonu
    Dense(1)        # Çıkış katmanı: tek değer (baraj doluluk oranı)
])
model.compile(optimizer='adam', loss='mse', metrics=['mae']) # Modeli derle: Adam optimizasyonu, kayıp fonksiyonu MSE, ek metrik MAE
model.summary() # Modelin yapısını ekrana yazdır

# ============================================================
# ADIM 9 - Model Eğitimi
# ============================================================
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True) 
# Eğer doğrulama kaybı 10 epoch boyunca iyileşmezse eğitim durur ve en iyi ağırlıklar geri yüklenir

history = model.fit(
    X_train_seq, y_train_seq,   # Eğitim verisi
    epochs=100,                 # Maksimum 100 epoch
    batch_size=32,              # Her seferde 32 örnek ile güncelleme
    validation_data=(X_test_seq, y_test_seq), # Doğrulama verisi
    callbacks=[early_stop],     # Erken durdurma kullan
    verbose=1                   # Eğitim sürecini ekrana yazdır
)
print("\nEğitim tamamlandı.") # Eğitim bitti mesajı

# ============================================================
# ADIM 10 - Tahmin ve Performans
# ============================================================
y_pred_scaled = model.predict(X_test_seq) # Test verisiyle tahmin yap (ölçeklenmiş değerler)
y_pred   = scaler_y.inverse_transform(y_pred_scaled) # Tahminleri orijinal ölçeğe çevir
y_actual = scaler_y.inverse_transform(y_test_seq)    # Gerçek değerleri orijinal ölçeğe çevir

# Performans metrikleri hesapla
mse_lstm  = mean_squared_error(y_actual, y_pred) # Ortalama kare hata
rmse_lstm = np.sqrt(mse_lstm)                    # Karekök ortalama kare hata
mae_lstm  = mean_absolute_error(y_actual, y_pred) # Ortalama mutlak hata
r2_lstm   = r2_score(y_actual, y_pred)            # R2 skoru

# Sonuçları tablo gibi yazdır
print("\n" + "="*52)
print("MODEL PERFORMANS KARŞILAŞTIRMASI")
print("="*52)
print(f"{'Model':<12} {'MSE':>8} {'RMSE':>8} {'MAE':>8} {'R2':>8}")
print("-"*52)
print(f"{'Naive':<12} {rmse_naive**2:>8.3f} {rmse_naive:>8.3f} {mae_naive:>8.3f} {r2_naive:>8.3f}")
print(f"{'LinReg':<12} {rmse_lr**2:>8.3f} {rmse_lr:>8.3f} {mae_lr:>8.3f} {r2_lr:>8.3f}")
print(f"{'LSTM':<12} {mse_lstm:>8.3f} {rmse_lstm:>8.3f} {mae_lstm:>8.3f} {r2_lstm:>8.3f}")
print("="*52)

# Gerçek vs Tahmin grafiği
plt.figure(figsize=(14, 5)) # Grafik boyutu
plt.plot(y_actual, label='Gerçek Doluluk (%)', linewidth=1.5) # Gerçek değerler
plt.plot(y_pred,   label='LSTM Tahmin (%)', linewidth=1.5, linestyle='--') # Tahmin edilen değerler
plt.title("Baraj Doluluk Oranı: Gerçek vs Tahmin") # Başlık
plt.xlabel("Gün") # X ekseni etiketi
plt.ylabel("Doluluk (%)") # Y ekseni etiketi
plt.legend() # Açıklama kutusu
plt.tight_layout() # Grafik düzenini sıkıştır
plt.show() # Grafiği göster

# Eğitim kayıp grafiği
plt.figure(figsize=(10, 4)) # Grafik boyutu
plt.plot(history.history['loss'],     label='Train Loss') # Eğitim kaybı
plt.plot(history.history['val_loss'], label='Validation Loss') # Doğrulama kaybı
plt.title("Eğitim Kayıp Grafiği") # Başlık
plt.xlabel("Epoch") # X ekseni etiketi
plt.ylabel("MSE")   # Y ekseni etiketi
plt.legend()        # Açıklama kutusu
plt.tight_layout()  # Grafik düzenini sıkıştır
plt.show()          # Grafiği göster
