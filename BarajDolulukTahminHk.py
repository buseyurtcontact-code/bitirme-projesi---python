# Gerekli kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
#  düzenlemeler yapılacak kod taslağı -> Denetimli Öğrenme + Geri Yayılım (Backpropagation) + LSTM

# VERİYİ YÜKLEME
df = pd.read_csv("izmir_baraj_veri.csv") # mail atıldı verilerin elime geçmesini bekliyorum 

df["date"] = pd.to_datetime(df["date"]) # tarih olduğu için datetime 

# Tarihe göre sırala
df = df.sort_values("date")

# EKSİK VERİ TEMİZLEME

df.interpolate(method="linear", inplace=True) # Küçük boşluklar için linear interpolation kullanacağız

# FEATURE ENGINEERING

# Lag özellikleri
df["lag1"] = df["doluluk"].shift(1)
df["lag7"] = df["doluluk"].shift(7)

# Moving average
df["ma7"] = df["doluluk"].rolling(window=7).mean()

# Ay bilgisi
df["month"] = df["date"].dt.month

# NA oluşan ilk satırları sil
df.dropna(inplace=True)

# TRAIN-TEST AYIRMA

train_size = int(len(df) * 0.8)

train = df[:train_size]
test = df[train_size:]

features = ["doluluk", "lag1", "lag7", "ma7", "month"]

X_train = train[features]
y_train = train["doluluk"]

X_test = test[features]
y_test = test["doluluk"]

#  NORMALİZASYON

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

y_train = scaler_y.fit_transform(y_train.values.reshape(-1,1))
y_test = scaler_y.transform(y_test.values.reshape(-1,1))

# BASELINE MODEL (LINEAR REGRESSION)

lr = LinearRegression()
lr.fit(X_train, y_train)

baseline_pred = lr.predict(X_test)

#  LSTM İÇİN SHAPE DÖNÜŞÜMÜ

# LSTM 3 boyutlu veri ister:
# (örnek sayısı, zaman adımı, özellik sayısı)

X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# LSTM MODELİ

model = Sequential()

model.add(LSTM(64, input_shape=(1, X_train.shape[1])))
model.add(Dropout(0.2))
model.add(Dense(32, activation="relu"))
model.add(Dense(1))

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

# Early stopping
early_stop = EarlyStopping(patience=10, restore_best_weights=True)

# MODEL EĞİTİMİ

history = model.fit(
    X_train_lstm,
    y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

#  TAHMİN

predictions = model.predict(X_test_lstm)

# Ölçek geri dönüşümü
predictions = scaler_y.inverse_transform(predictions)
y_test_original = scaler_y.inverse_transform(y_test)

# PERFORMANS

rmse = np.sqrt(mean_squared_error(y_test_original, predictions))
mae = mean_absolute_error(y_test_original, predictions)
r2 = r2_score(y_test_original, predictions)

print("RMSE:", rmse)
print("MAE:", mae)
print("R2:", r2)

# GRAFİK

plt.figure(figsize=(10,5))
plt.plot(y_test_original, label="Gerçek")
plt.plot(predictions, label="Tahmin")
plt.legend()
plt.title("Gerçek vs Tahmin Doluluk Oranı")
plt.show()
