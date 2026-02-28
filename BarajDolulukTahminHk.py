import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# -------------------------------
# deneme için rastgele veri üretmek

np.random.seed(42)
n_days = 365
dates = pd.date_range(start="2025-01-01", periods=n_days)

# Rastgele Mevsimsel trend (yaz-kış)
seasonal_trend = 50 + 20*np.sin(2*np.pi*np.arange(n_days)/365)

# Rastgele Haftalık döngü (hafta sonları biraz farklı) düzenlenecek
weekly_cycle = 2 * np.sin(2*np.pi*np.arange(n_days)/7)

# Rastgele yağış/kuraklık etkisi
random_noise = np.random.normal(0, 5, n_days)

# Rastgele Doluluk oranı
doluluk = seasonal_trend + weekly_cycle + random_noise
df = pd.DataFrame({"date": dates, "doluluk": doluluk})

# VERİ TEMİZLEME VE FEATURE ENGINEERING

#  Küçük boşluklar (1-2 gün) için doğrusal interpolasyon
df['doluluk_linear'] = df['doluluk'].interpolate(method='linear', limit=2)

#  Büyük boşluklar (3+ gün) için mevsimsel interpolasyon
# Mevsimsel trend
seasonal_trend = 50 + 20*np.sin(2*np.pi*np.arange(n_days)/365)

df["daily_change"] = df["doluluk"].diff()
max_daily_change = 10
min_daily_change = -10
df["doluluk_cleaned"] = df["doluluk"].copy()
df.loc[df["daily_change"] > max_daily_change, "doluluk_cleaned"] = df["doluluk"].shift(1) + max_daily_change
df.loc[df["daily_change"] < min_daily_change, "doluluk_cleaned"] = df["doluluk"].shift(1) + min_daily_change

# Lag ve hareketli ortalama
df["lag1"] = df["doluluk_cleaned"].shift(1)
df["lag7"] = df["doluluk_cleaned"].shift(7)
df["ma7"] = df["doluluk_cleaned"].rolling(window=7).mean()
df["month"] = df["date"].dt.month
df["season"] = df["month"] % 12 // 3 + 1
df.dropna(inplace=True)

features = ["lag1", "lag7", "ma7", "month", "season", "daily_change"]
target = "doluluk_cleaned"

# -------------------------------
# SLIDING WINDOW
# -------------------------------
def create_sliding_window(data, features, target, window_size=7):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[features].iloc[i:i+window_size].values)
        y.append(data[target].iloc[i+window_size])
    return np.array(X), np.array(y).reshape(-1,1)

window_size = 7
X, y = create_sliding_window(df, features, target, window_size)

# -------------------------------
# NORMALİZASYON
# -------------------------------
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
n_samples, n_timesteps, n_features = X.shape
X_reshaped = X.reshape(n_samples, n_timesteps*n_features)
X_scaled = scaler_X.fit_transform(X_reshaped).reshape(n_samples, n_timesteps, n_features)
y_scaled = scaler_y.fit_transform(y)

# -------------------------------
# TRAIN-TEST AYIRMA
# -------------------------------
train_size = int(len(X_scaled) * 0.8)
X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
y_train, y_test = y_scaled[:train_size], y_scaled[train_size:]

# -------------------------------
# LSTM MODELİ
# -------------------------------
model = Sequential()
model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(32, activation="relu"))
model.add(Dense(1))
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

early_stop = EarlyStopping(patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

# -------------------------------
# TAHMİN VE METRİKLER
# -------------------------------
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_original = scaler_y.inverse_transform(y_test)

rmse = np.sqrt(mean_squared_error(y_test_original, y_pred))
mae = mean_absolute_error(y_test_original, y_pred)
r2 = r2_score(y_test_original, y_pred)

print("RMSE:", rmse)
print("MAE:", mae)
print("R2:", r2)

# -------------------------------
# GRAFİK
# -------------------------------
plt.figure(figsize=(12,5))
plt.plot(y_test_original, label="Gerçek")
plt.plot(y_pred, label="Tahmin")
plt.legend()
plt.title("Gerçek vs Tahmin Doluluk Oranı (Simülasyon Verisi)")
plt.show()
