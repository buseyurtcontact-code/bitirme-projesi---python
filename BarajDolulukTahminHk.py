# -*- coding: utf-8 -*-   # Türkçe karakterlerin sorunsuz çalışması için dosya kodlamasını ayarlıyoruz.

import numpy as np        # Sayısal hesaplamalar için kullanılan kütüphane.
import pandas as pd       # Veri analizi ve tablo işlemleri için kullanılan kütüphane.
import matplotlib.pyplot as plt  # Grafik çizmek için kullanılan kütüphane.
import seaborn as sns     # Daha şık grafikler çizmek için kullanılan kütüphane.
from sklearn.preprocessing import MinMaxScaler  # Verileri ölçeklemek için.
from sklearn.linear_model import LinearRegression  # Basit doğrusal regresyon modeli için.
import tensorflow as tf   # Derin öğrenme için kullanılan kütüphane.
from tensorflow.keras.models import Sequential   # Yapay sinir ağı modeli kurmak için.
from tensorflow.keras.layers import LSTM, Dense, Dropout  # LSTM ve diğer katmanlar için.

# dosya yolu
data = pd.read_csv(r"C:\Users\acer\Desktop\310.py\tezzz.py\seattle-weather.csv")  
# CSV dosyasını bilgisayardan okuyoruz ve 'data' isimli tabloya yüklüyoruz.

data['date'] = pd.to_datetime(data['date'])  
# 'date' sütununu tarih formatına çeviriyoruz ki üzerinde zaman işlemleri yapabilelim.

data = data.sort_values('date')  
# Verileri tarihe göre sıralıyoruz (eskiden yeniye doğru).

# Girdi (X) ve çıktı (y) ayır
X = data[['precipitation', 'temp_max', 'temp_min', 'wind']]  
# Modelin kullanacağı giriş verileri: yağış, maksimum sıcaklık, minimum sıcaklık ve rüzgar.
y = data['weather']  
# Çıkış verisi: hava durumu (örneğin 'rain', 'drizzle', 'sun').

'''
çıktısı:
   precipitation  temp_max  temp_min  wind precipitation=yağış miktarı
0            0.0      12.8       5.0   4.7
1           10.9      10.6       2.8   4.5
2            0.8      11.7       7.2   2.3
3           20.3      12.2       5.6   4.7
4            1.3       8.9       2.8   6.1
0    drizzle
1       rain
2       rain
3       rain
4       rain
Name: weather, dtype: object
'''
# Yukarıdaki örnek çıktıda X tablosu sayısal değerleri, y ise hava durumu etiketlerini gösteriyor.

# Küçük boşluklar için lineer interpolasyon
data[['precipitation','temp_max','temp_min','wind']] = (
    data[['precipitation','temp_max','temp_min','wind']].interpolate(method='linear')
)
# Eğer verilerde küçük eksikler varsa, onları doğrusal (lineer) yöntemle dolduruyoruz.

# Büyük boşluklar için mevsimsel ortalama doldurma
data['month'] = data['date'].dt.month  
# Her satır için ay bilgisini çıkarıyoruz.
for col in ['precipitation','temp_max','temp_min','wind']:
    data[col] = data.groupby('month')[col].transform(lambda x: x.fillna(x.mean()))
# Eğer büyük boşluklar varsa, aynı ayın ortalama değerleriyle dolduruyoruz.
# Örneğin Ocak ayındaki eksik değerler Ocak ortalamasıyla tamamlanıyor.

# Güncel X ve y tekrar tanımla
X = data[['precipitation', 'temp_max', 'temp_min', 'wind']]  
# Eksikleri doldurduktan sonra giriş verilerini tekrar tanımlıyoruz.
y = data['weather']  
# Çıkış verisini tekrar tanımlıyoruz.

print(X.head())  
# İlk 5 satırdaki giriş verilerini ekrana yazdırıyoruz.
print(y.head())  
# İlk 5 satırdaki çıkış verilerini ekrana yazdırıyoruz.
print(data.head(20))  
# Tablonun ilk 20 satırını ekrana yazdırıyoruz.
