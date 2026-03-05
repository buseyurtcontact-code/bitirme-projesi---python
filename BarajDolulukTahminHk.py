# -*- coding: utf-8 -*-   # Türkçe karakterlerin sorunsuz çalışması için dosya kodlamasını ayarlıyoruz.
# İzmir baraj doluluk oranı tahmini için örnek LSTM kodu
# Şimdilik Seattle verisiyle yağış tahmini yapıyoruz (sayısal hedef)

import numpy as np        # Sayısal hesaplamalar için kullanılan kütüphane.
import pandas as pd       # Veri analizi ve tablo işlemleri için kullanılan kütüphane.
import matplotlib.pyplot as plt  # Grafik çizmek için kullanılan kütüphane.
from sklearn.preprocessing import MinMaxScaler  # Verileri ölçeklemek için.
from sklearn.model_selection import train_test_split  # Eğitim ve test verisini ayırmak için.
from tensorflow.keras.models import Sequential   # Yapay sinir ağı modeli kurmak için.
from tensorflow.keras.layers import LSTM, Dense, Dropout  # LSTM ve diğer katmanlar için.

# 1. Veri yükleme
data = pd.read_csv(r"C:\Users\acer\Desktop\310.py\tezzz.py\seattle-weather.csv")  # CSV dosyasını okuyoruz.
data['date'] = pd.to_datetime(data['date'])   # 'date' sütununu tarih formatına çeviriyoruz.
data = data.sort_values('date').reset_index(drop=True)  # Verileri tarihe göre sıralıyoruz ve indexleri sıfırlıyoruz.

# 2. Eksik veri doldurma
data['month'] = data['date'].dt.month  # Her satır için ay bilgisini çıkarıyoruz.
for col in ['precipitation', 'temp_max', 'temp_min', 'wind']:
    data[col] = data[col].interpolate(method='linear')  # Küçük boşlukları lineer interpolasyon ile dolduruyoruz.
    data[col] = data.groupby('month')[col].transform(lambda x: x.fillna(x.mean()))  # Büyük boşlukları aynı ayın ortalamasıyla dolduruyoruz.

# 3. Girdi ve çıktı değişkenleri
# ⚠️ Gerçek projede hedef: data['reservoir_percent'] olacak.
# Şimdilik yağışı tahmin ediyoruz.
X = data[['temp_max', 'temp_min', 'wind']].values  # Girdi değişkenleri: sıcaklıklar ve rüzgar.
y = data['precipitation'].values  # Çıkış değişkeni: yağış miktarı.

# 4. Ölçekleme
scaler_X = MinMaxScaler()  # Girdi verilerini ölçeklemek için.
scaler_y = MinMaxScaler()  # Çıkış verisini ölçeklemek için.
X_scaled = scaler_X.fit_transform(X)  # Girdi verilerini 0-1 aralığına ölçekliyoruz.
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))  # Çıkış verisini 0-1 aralığına ölçekliyoruz.

# 5. Zaman serisi için pencere oluşturma (ör. son 30 gün)
WINDOW = 30  # Kaç gün geriye bakılacağını belirliyoruz.
def create_sequences(X, y, window):
    Xs, ys = [], []
    for i in range(len(X) - window):
        Xs.append(X[i:i+window])   # Son 30 günün verilerini giriş olarak alıyoruz.
        ys.append(y[i+window])     # 31. günün değerini çıkış olarak alıyoruz.
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_sequences(X_scaled, y_scaled, WINDOW)  # Fonksiyonu çalıştırıyoruz.
# X_seq.shape → (örnek sayısı, 30 gün, 3 özellik)

# 6. Train/test split (shuffle=False → zaman serisi bozulmasın)
X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq, test_size=0.2, shuffle=False  # Verilerin %80'i eğitim, %20'si test için ayrılıyor.
)

# 7. LSTM Modeli
model = Sequential([  # Modeli sırayla katmanlar ekleyerek tanımlıyoruz.
    LSTM(64, return_sequences=True, input_shape=(WINDOW, X.shape[1])),  # İlk LSTM katmanı, 64 nöron.
    Dropout(0.2),  # %20 dropout → aşırı öğrenmeyi azaltmak için.
    LSTM(32),  # İkinci LSTM katmanı, 32 nöron.
    Dense(1)   # Çıkış katmanı → tek değer (yağış veya baraj doluluk %).
])
model.compile(optimizer='adam', loss='mse')  # Modeli derliyoruz, kayıp fonksiyonu MSE.
model.summary()  # Modelin yapısını ekrana yazdırıyoruz.

# 8. Model eğitimi
history = model.fit(
    X_train, y_train,  # Eğitim verisi
    epochs=50,         # 50 kez tüm veri üzerinde eğitim yapılacak.
    batch_size=32,     # Her seferde 32 örnek kullanılacak.
    validation_data=(X_test, y_test)  # Test verisi doğrulama için kullanılacak.
)

# 9. Tahmin
y_pred_scaled = model.predict(X_test)  # Test verisi üzerinde tahmin yapıyoruz.
y_pred = scaler_y.inverse_transform(y_pred_scaled)  # Tahminleri ölçekten geri çeviriyoruz.
y_actual = scaler_y.inverse_transform(y_test)  # Gerçek değerleri ölçekten geri çeviriyoruz.

# 10. Görselleştirme
plt.figure(figsize=(12, 6))  # Grafik boyutunu ayarlıyoruz.
plt.plot(y_actual, label='Gerçek')  # Gerçek değerleri çiziyoruz.
plt.plot(y_pred, label='Tahmin')    # Tahmin edilen değerleri çiziyoruz.
plt.legend()  # Grafik üzerine açıklama ekliyoruz.
plt.title("Yağış Tahmini (Gerçek Projede: Baraj Doluluk %)")  # Başlık ekliyoruz.
plt.show()  # Grafiği ekrana gösteriyoruz.
