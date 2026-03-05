# -*- coding: utf-8 -*-
# Izmir Baraj Doluluk Orani Tahmini - LSTM Modeli
# Simdilik Seattle verisiyle calisiyoruz (sentetik reservoir_percent ile)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ============================================================
# ADIM 1 - Veri Yukleme
# ============================================================
data = pd.read_csv(r"C:\Users\acer\Desktop\310.py\tezzz.py\seattle-weather.csv")
data['date'] = pd.to_datetime(data['date'])
data = data.sort_values('date').reset_index(drop=True)
print("Veri yuklendi. Satir sayisi:", len(data))

# Sentetik baraj doluluk orani
# Gercek veri geldiginde bu blogu silip CSV'den oku
data['reservoir_percent'] = (
    50
    + 0.3 * data['precipitation']
    - 0.1 * data['temp_max']
    + 0.05 * data['wind']
).clip(0, 100)

# ============================================================
# ADIM 2 - Eksik Veri Doldurma
# ============================================================
data['month'] = data['date'].dt.month
for col in ['reservoir_percent', 'precipitation', 'temp_max', 'temp_min', 'wind']:
    data[col] = data[col].interpolate(method='linear')
    data[col] = data.groupby('month')[col].transform(lambda x: x.fillna(x.mean()))

# Aykirlik raporu
print("\nAykirlik raporu:")
for col in ['precipitation', 'temp_max', 'temp_min', 'wind', 'reservoir_percent']:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    n = ((data[col] < Q1 - 1.5 * IQR) | (data[col] > Q3 + 1.5 * IQR)).sum()
    print(f"  {col}: {n} aykiri deger")

# ============================================================
# ADIM 3 - Feature Engineering
# ============================================================
target = 'reservoir_percent'

data['lag_1']    = data[target].shift(1)
data['lag_7']    = data[target].shift(7)
data['lag_30']   = data[target].shift(30)
data['rolling_7']= data[target].rolling(window=7).mean()
data['season']   = data['month'].map({
    12:1, 1:1, 2:1,
    3:2,  4:2, 5:2,
    6:3,  7:3, 8:3,
    9:4, 10:4, 11:4
})

data = data.dropna().reset_index(drop=True)

features = ['precipitation', 'temp_max', 'temp_min', 'wind',
            'lag_1', 'lag_7', 'lag_30', 'rolling_7', 'month', 'season']

X_raw = data[features].values
y_raw = data[target].values
print("\nOzellik sayisi:", len(features))
print("Toplam ornek sayisi:", len(X_raw))

# ============================================================
# ADIM 4 - Train-Test Bolme (shuffle=False)
# ============================================================
split = int(len(X_raw) * 0.8)
X_train_raw, X_test_raw = X_raw[:split], X_raw[split:]
y_train_raw, y_test_raw = y_raw[:split], y_raw[split:]
print(f"\nTrain: {len(X_train_raw)} satir | Test: {len(X_test_raw)} satir")

# ============================================================
# ADIM 5 - Normalizasyon (scaler sadece train ile fit edilir)
# ============================================================
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_scaled = scaler_X.fit_transform(X_train_raw)
X_test_scaled  = scaler_X.transform(X_test_raw)

y_train_scaled = scaler_y.fit_transform(y_train_raw.reshape(-1, 1))
y_test_scaled  = scaler_y.transform(y_test_raw.reshape(-1, 1))

# ============================================================
# ADIM 6 - Baseline Modeller
# ============================================================
# Naive: yarin = bugun
y_naive        = y_test_raw[:-1]
y_naive_actual = y_test_raw[1:]
rmse_naive = np.sqrt(mean_squared_error(y_naive_actual, y_naive))
mae_naive  = mean_absolute_error(y_naive_actual, y_naive)
r2_naive   = r2_score(y_naive_actual, y_naive)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train_raw)
y_lr_pred = lr.predict(X_test_scaled)
rmse_lr = np.sqrt(mean_squared_error(y_test_raw, y_lr_pred))
mae_lr  = mean_absolute_error(y_test_raw, y_lr_pred)
r2_lr   = r2_score(y_test_raw, y_lr_pred)

# ============================================================
# ADIM 7 - Sliding Window (gecmis 30 gun -> 31. gunu tahmin et)
# ============================================================
WINDOW = 30

def create_sequences(X, y, window):
    Xs, ys = [], []
    for i in range(len(X) - window):
        Xs.append(X[i:i + window])
        ys.append(y[i + window])
    return np.array(Xs), np.array(ys)

X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, WINDOW)
X_test_seq,  y_test_seq  = create_sequences(X_test_scaled,  y_test_scaled,  WINDOW)
print(f"\nSequence shape: {X_train_seq.shape}")

# ============================================================
# ADIM 8 - LSTM Model Tasarimi
# ============================================================
model = Sequential([
    LSTM(64, return_sequences=False, input_shape=(WINDOW, len(features))),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# ============================================================
# ADIM 9 - Model Egitimi
# ============================================================
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train_seq, y_train_seq,
    epochs=100,
    batch_size=32,
    validation_data=(X_test_seq, y_test_seq),
    callbacks=[early_stop],
    verbose=1
)
print("\nEgitim tamamlandi.")

# ============================================================
# ADIM 10 - Tahmin ve Performans
# ============================================================
y_pred_scaled = model.predict(X_test_seq)
y_pred   = scaler_y.inverse_transform(y_pred_scaled)
y_actual = scaler_y.inverse_transform(y_test_seq)

mse_lstm  = mean_squared_error(y_actual, y_pred)
rmse_lstm = np.sqrt(mse_lstm)
mae_lstm  = mean_absolute_error(y_actual, y_pred)
r2_lstm   = r2_score(y_actual, y_pred)

print("\n" + "="*52)
print("MODEL PERFORMANS KARSILASTIRMASI")
print("="*52)
print(f"{'Model':<12} {'MSE':>8} {'RMSE':>8} {'MAE':>8} {'R2':>8}")
print("-"*52)
print(f"{'Naive':<12} {rmse_naive**2:>8.3f} {rmse_naive:>8.3f} {mae_naive:>8.3f} {r2_naive:>8.3f}")
print(f"{'LinReg':<12} {rmse_lr**2:>8.3f} {rmse_lr:>8.3f} {mae_lr:>8.3f} {r2_lr:>8.3f}")
print(f"{'LSTM':<12} {mse_lstm:>8.3f} {rmse_lstm:>8.3f} {mae_lstm:>8.3f} {r2_lstm:>8.3f}")
print("="*52)

# Gercek vs Tahmin grafigi
plt.figure(figsize=(14, 5))
plt.plot(y_actual, label='Gercek Doluluk (%)', linewidth=1.5)
plt.plot(y_pred,   label='LSTM Tahmin (%)', linewidth=1.5, linestyle='--')
plt.title("Baraj Doluluk Orani: Gercek vs Tahmin")
plt.xlabel("Gun")
plt.ylabel("Doluluk (%)")
plt.legend()
plt.tight_layout()
plt.show()

# Egitim kayip grafigi
plt.figure(figsize=(10, 4))
plt.plot(history.history['loss'],     label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Egitim Kayip Grafigi")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.legend()
plt.tight_layout()
plt.show()
