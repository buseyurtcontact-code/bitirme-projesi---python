# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

import pandas as pd

# Doğru dosya yolu
data = pd.read_csv(r"C:\Users\acer\Desktop\310.py\tezzz.py\seattle-weather.csv")

# Tarih kolonunu datetime formatına çevir
data['date'] = pd.to_datetime(data['date'])

# Tarihe göre sırala
data = data.sort_values('date')

# Girdi (X) ve çıktı (y) ayır
X = data[['precipitation', 'temp_max', 'temp_min', 'wind']]
y = data['weather']
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

# Küçük boşluklar için lineer interpolasyon
data[['precipitation','temp_max','temp_min','wind']] = (
    data[['precipitation','temp_max','temp_min','wind']].interpolate(method='linear')
)

# Büyük boşluklar için mevsimsel ortalama doldurma
data['month'] = data['date'].dt.month
for col in ['precipitation','temp_max','temp_min','wind']:
    data[col] = data.groupby('month')[col].transform(lambda x: x.fillna(x.mean()))

# Güncel X ve y tekrar tanımla
X = data[['precipitation', 'temp_max', 'temp_min', 'wind']]
y = data['weather']

print(X.head())
print(y.head())
print(data.head(20))
  
