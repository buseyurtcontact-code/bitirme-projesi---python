# -*- coding: utf-8 -*-
# Hava Durumu Sınıflandırması - LSTM Modeli (v3 - Dengesizlik Düzeltildi)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.optimizers import Adam

# ============================================================
# ADIM 1 - Veri Yükleme
# ============================================================
data = pd.read_csv(r"C:\Users\acer\Desktop\310.py\tezzz.py\seattle-weather.csv")
data['date'] = pd.to_datetime(data['date'])
data = data.sort_values('date').reset_index(drop=True)

print("Veri yüklendi. Satır sayısı:", len(data))
print("\nHava durumu sınıfları ve dağılımı:")
print(data['weather'].value_counts())

# ============================================================
# ADIM 2 - Özellikler
# ============================================================
data['month']      = data['date'].dt.month
data['season']     = data['date'].dt.month % 12 // 3
data['temp_range'] = data['temp_max'] - data['temp_min']   # YENİ: sıcaklık farkı
data['precip_log'] = np.log1p(data['precipitation'])       # YENİ: yağış log dönüşümü

FEATURES = ['precipitation', 'precip_log', 'temp_max', 'temp_min',
            'temp_range', 'wind', 'month', 'season']
TARGET   = 'weather'

for col in FEATURES:
    data[col] = data[col].interpolate(method='linear')
    data[col] = data[col].fillna(data[col].mean())

data = data[FEATURES + [TARGET]].dropna().reset_index(drop=True)
print(f"\nKullanılan özellikler : {FEATURES}")
print(f"Toplam temiz örnek    : {len(data)}")

# ============================================================
# ADIM 3 - Encode
# ============================================================
le = LabelEncoder()
data['label'] = le.fit_transform(data[TARGET])
num_classes = len(le.classes_)
print(f"\nSınıf eşlemesi: { {c: int(le.transform([c])[0]) for c in le.classes_} }")
print(f"Toplam sınıf sayısı: {num_classes}")

# ============================================================
# ADIM 4 - Train/Test Bölme
# ============================================================
X_raw = data[FEATURES].values
y_raw = data['label'].values

split = int(len(X_raw) * 0.8)
X_train_raw, X_test_raw = X_raw[:split], X_raw[split:]
y_train_raw, y_test_raw = y_raw[:split], y_raw[split:]
print(f"\nTrain: {len(X_train_raw)} satır  |  Test: {len(X_test_raw)} satır")

# ============================================================
# ADIM 5 - Normalizasyon
# ============================================================
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled  = scaler.transform(X_test_raw)

# ============================================================
# ADIM 6 - Baseline Modeller
# ============================================================
dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_train_scaled, y_train_raw)
acc_dummy = accuracy_score(y_test_raw, dummy.predict(X_test_scaled))

lr = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')
lr.fit(X_train_scaled, y_train_raw)
acc_lr = accuracy_score(y_test_raw, lr.predict(X_test_scaled))

print(f"\nBaseline (En Sık Sınıf) Doğruluk : {acc_dummy:.4f}")
print(f"Lojistik Regresyon Doğruluk       : {acc_lr:.4f}")

# ============================================================
# ADIM 7 - Sliding Window  (30 → 14)
# ============================================================
WINDOW = 14   # ✅ Küçültüldü: daha az veri kaybı, daha fazla örnek
def create_sequences(X, y, window):
    Xs, ys = [], []
    for i in range(len(X) - window):
        Xs.append(X[i : i + window])
        ys.append(y[i + window])
    return np.array(Xs), np.array(ys)

X_train_seq, y_train_seq_raw = create_sequences(X_train_scaled, y_train_raw, WINDOW)
X_test_seq,  y_test_seq_raw  = create_sequences(X_test_scaled,  y_test_raw,  WINDOW)

print(f"\nSequence shape (train) : {X_train_seq.shape}")

# ============================================================
# ADIM 8 - SMOTE ile Azınlık Sınıflarını Dengele
# ============================================================
# Sequence verisini 2D'ye çevir, SMOTE uygula, geri al
n_train, w, f = X_train_seq.shape
X_train_2d = X_train_seq.reshape(n_train, w * f)

print("\nSMOTE öncesi sınıf dağılımı:")
unique, counts = np.unique(y_train_seq_raw, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  {le.classes_[u]}: {c}")

smote = SMOTE(random_state=42, k_neighbors=3)
X_train_res, y_train_res = smote.fit_resample(X_train_2d, y_train_seq_raw)
X_train_seq_res = X_train_res.reshape(-1, w, f)

print("\nSMOTE sonrası sınıf dağılımı:")
unique, counts = np.unique(y_train_res, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  {le.classes_[u]}: {c}")

# Kategorik dönüşüm
y_train_cat = to_categorical(y_train_res,      num_classes=num_classes)
y_test_cat  = to_categorical(y_test_seq_raw,   num_classes=num_classes)
y_test_int  = y_test_seq_raw

# ============================================================
# ADIM 9 - LSTM Model  (BatchNormalization eklendi)
# ============================================================
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(WINDOW, len(FEATURES))),
    BatchNormalization(),
    Dropout(0.3),
    LSTM(64, return_sequences=False),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy', Precision(name='precision'), Recall(name='recall')]
)
model.summary()

# ============================================================
# ADIM 10 - Eğitim  (ReduceLROnPlateau eklendi)
# ============================================================
# SMOTE sonrası class_weight gerekmez ama yine de hafif tutalım
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_res),
    y=y_train_res
)
class_weights = dict(enumerate(class_weights))

early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr  = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)

history = model.fit(
    X_train_seq_res, y_train_cat,
    epochs=100,
    batch_size=32,
    validation_data=(X_test_seq, y_test_cat),
    callbacks=[early_stop, reduce_lr],
    class_weight=class_weights,
    verbose=1
)
print("\nEğitim tamamlandı.")

# ============================================================
# ADIM 11 - Tahmin & Metrikler
# ============================================================
y_pred_proba  = model.predict(X_test_seq)
y_pred_int    = np.argmax(y_pred_proba, axis=1)
y_pred_labels = le.inverse_transform(y_pred_int)
y_true_labels = le.inverse_transform(y_test_int)

acc_lstm = accuracy_score(y_test_int, y_pred_int)
auc_lstm = roc_auc_score(y_test_cat, y_pred_proba, multi_class='ovr')

print("\n" + "="*52)
print("       MODEL PERFORMANS KARŞILAŞTIRMASI")
print("="*52)
print(f"{'Model':<28} {'Accuracy':>12}")
print("-"*52)
print(f"{'Baseline (En Sık Sınıf)':<28} {acc_dummy:>12.4f}")
print(f"{'Lojistik Regresyon':<28} {acc_lr:>12.4f}")
print(f"{'LSTM (v3)':<28} {acc_lstm:>12.4f}")
print("="*52)
print(f"LSTM ROC-AUC: {auc_lstm:.4f}")

print("\nDetaylı Sınıflandırma Raporu (LSTM v3):")
present_classes = np.unique(np.concatenate([y_true_labels, y_pred_labels]))
print(classification_report(y_true_labels, y_pred_labels,
                             labels=present_classes,
                             target_names=present_classes))

# ============================================================
# ADIM 12 - Görselleştirmeler
# ============================================================

# --- 1) Confusion Matrix ---
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_true_labels, y_pred_labels, labels=present_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=present_classes, yticklabels=present_classes)
plt.title("Confusion Matrix – LSTM v3 Hava Durumu Sınıflandırması")
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek")
plt.tight_layout()
plt.show()

# --- 2) Loss & Accuracy Grafikleri ---
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].plot(history.history['loss'],     label='Train Loss')
axes[0].plot(history.history['val_loss'], label='Validation Loss')
axes[0].set_title("Kayıp Grafiği"); axes[0].legend()

axes[1].plot(history.history['accuracy'],     label='Train Accuracy')
axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
axes[1].set_title("Doğruluk Grafiği"); axes[1].legend()
plt.tight_layout()
plt.show()

# --- 3) Histogram: Gerçek vs Tahmin ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(y_true_labels, bins=len(present_classes),
             color='steelblue', edgecolor='white', linewidth=0.8, rwidth=0.85)
axes[0].set_xticks(range(len(present_classes)))
axes[0].set_xticklabels(present_classes, rotation=15)
axes[0].set_title("Gerçek Sınıf Dağılımı")
axes[0].set_xlabel("Hava Durumu Sınıfı")
axes[0].set_ylabel("Örnek Sayısı")

axes[1].hist(y_pred_labels, bins=len(present_classes),
             color='darkorange', edgecolor='white', linewidth=0.8, rwidth=0.85)
axes[1].set_xticks(range(len(present_classes)))
axes[1].set_xticklabels(present_classes, rotation=15)
axes[1].set_title("Tahmin Edilen Sınıf Dağılımı")
axes[1].set_xlabel("Hava Durumu Sınıfı")
axes[1].set_ylabel("Örnek Sayısı")

plt.suptitle("LSTM v3 – Gerçek vs Tahmin Edilen Sınıf Histogramları",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
