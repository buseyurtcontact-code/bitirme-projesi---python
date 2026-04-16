'''

XOR problemi 

x1	x2	çıktı
0	0	0
0	1	1
1	0	1
1	1	0

x1 ve x2 verdiğimizde çıktıyı öğretmeye çalışıyoruz

'''


import math
import random

# ─────────────────────────────────────────────────────────────────
#  Aktivasyon Fonksiyonları
#  XOR gibi doğrusal olmayan problemlerde aktivasyon şarttır.
#  Sigmoid çıktıyı [0, 1] aralığına sıkıştırır.
# ─────────────────────────────────────────────────────────────────

def sigmoid(x):
    """Sigmoid aktivasyon: f(x) = 1 / (1 + e^(-x))"""
    return 1.0 / (1.0 + math.exp(-x))

def sigmoid_turev(y):
    """
    Sigmoid'in türevi: f'(x) = f(x) · (1 − f(x))
    Geri yayılımda kullanılır. y = sigmoid(x) olduğu varsayılır,
    tekrar hesaplamamak için sigmoid çıktısını direkt alıyoruz.
    """
    return y * (1.0 - y)


# ─────────────────────────────────────────────────────────────────
#  Çok Katmanlı Algılayıcı (MLP)
#  Mimari: 2 giriş → 2 gizli nöron → 1 çıkış
#
#  Neden 2 gizli nöron?
#  XOR = (x1 OR x2) AND NOT (x1 AND x2)
#  İki gizli nöron bu iki koşulu ayrı ayrı öğrenebilir.
# ─────────────────────────────────────────────────────────────────

class MLP:
    def __init__(self, n_giris=2, n_gizli=2, n_cikis=1, ogrenme_hizi=0.5):
        """
        Parametreler:
          n_giris       : giriş katmanındaki nöron sayısı
          n_gizli       : gizli katmandaki nöron sayısı
          n_cikis       : çıkış katmanındaki nöron sayısı
          ogrenme_hizi  : η — her güncellemede ne kadar adım atılacağı
        """
        self.lr = ogrenme_hizi

        # ── Ağırlıklar küçük rastgele değerlerle başlatılır ───────
        # Sıfırdan başlatmak tüm nöronları aynı hata sinyaline
        # götürür (simetri sorunu), bu yüzden rassal başlatma yapıyoruz.

        # Giriş → Gizli ağırlıkları:  W1[gizli_idx][giris_idx]
        self.W1 = [[random.uniform(-1, 1) for _ in range(n_giris)]
                   for _ in range(n_gizli)]

        # Gizli katman önyargıları (bias): her nöron için bir adet
        self.b1 = [random.uniform(-1, 1) for _ in range(n_gizli)]

        # Gizli → Çıkış ağırlıkları: W2[cikis_idx][gizli_idx]
        self.W2 = [[random.uniform(-1, 1) for _ in range(n_gizli)]
                   for _ in range(n_cikis)]

        # Çıkış katmanı önyargıları
        self.b2 = [random.uniform(-1, 1) for _ in range(n_cikis)]

    # ── İleri Besleme (Forward Pass) ──────────────────────────────
    def ileri_besleme(self, x):
        """
        Girişi ağdan geçirip çıktıyı hesaplar.

        Adımlar:
          1. net_h  = W1 · x + b1          (doğrusal kombinasyon)
          2. h      = sigmoid(net_h)         (aktivasyon)
          3. net_o  = W2 · h + b2
          4. cikis  = sigmoid(net_o)
        """
        # Gizli katman aktivasyonlarını hesapla
        self.h_net = []   # aktivasyon öncesi değerler (net girdi)
        self.h_out = []   # aktivasyon sonrası değerler

        for i, (agirliklar, bias) in enumerate(zip(self.W1, self.b1)):
            # Her gizli nöronun net girdisi: ağırlıklı toplam + bias
            net = sum(w * xi for w, xi in zip(agirliklar, x)) + bias
            self.h_net.append(net)
            self.h_out.append(sigmoid(net))   # sigmoid uygula

        # Çıkış katmanı aktivasyonlarını hesapla
        self.o_net = []
        self.o_out = []

        for agirliklar, bias in zip(self.W2, self.b2):
            net = sum(w * hi for w, hi in zip(agirliklar, self.h_out)) + bias
            self.o_net.append(net)
            self.o_out.append(sigmoid(net))

        return self.o_out   # çıkış vektörü

    # ── Geri Yayılım (Backpropagation) ───────────────────────────
    def geri_yayilim(self, x, hedef):
        """
        Zincir kuralıyla hata gradyanlarını hesaplar ve ağırlıkları günceller.

        Kayıp fonksiyonu: L = ½ Σ (hedef − çıkış)²

        Türevler (zincir kuralı):
          ∂L/∂W2 = ∂L/∂o · ∂o/∂net_o · ∂net_o/∂W2
                 = (o − t) · sigmoid'(o) · h

          ∂L/∂W1 = ∂L/∂h · sigmoid'(h) · x
          ∂L/∂h  = Σ [ (o − t) · sigmoid'(o) · W2 ]   (gizliye gelen hata)
        """
        # ── 1. Çıkış katmanı delta'sı ──────────────────────────
        # delta_o = (tahmin − hedef) · sigmoid'(tahmin)
        delta_o = []
        for tahmin, t in zip(self.o_out, hedef):
            hata   = tahmin - t
            delta  = hata * sigmoid_turev(tahmin)
            delta_o.append(delta)

        # ── 2. Gizli katman delta'sı ───────────────────────────
        # Hatayı gizli katmana taşı: her gizli nöronun çıkış
        # katmanına katkısı kadar hata alır.
        delta_h = []
        for j, h_aktivasyon in enumerate(self.h_out):
            # Bu gizli nörondan gelen tüm çıkış hatalarını topla
            gelen_hata = sum(delta_o[k] * self.W2[k][j]
                             for k in range(len(self.W2)))
            delta_h.append(gelen_hata * sigmoid_turev(h_aktivasyon))

        # ── 3. Ağırlıkları güncelle ────────────────────────────
        # W ← W − η · delta · önceki_katman_aktivasyonu

        # Çıkış katmanı güncelleme
        for k in range(len(self.W2)):
            for j in range(len(self.h_out)):
                self.W2[k][j] -= self.lr * delta_o[k] * self.h_out[j]
            self.b2[k] -= self.lr * delta_o[k]

        # Gizli katman güncelleme
        for j in range(len(self.W1)):
            for i, xi in enumerate(x):
                self.W1[j][i] -= self.lr * delta_h[j] * xi
            self.b1[j] -= self.lr * delta_h[j]

    # ── Eğitim Döngüsü ─────────────────────────────────────────
    def egit(self, X, Y, epochs=10000):
        """
        Veri setini epoch sayısı kadar tekrar eder.
        Her epoch'ta tüm örnekler üzerinden ileri besleme +
        geri yayılım uygulanır (tam toplu veya stokastik SGD).
        """
        for epoch in range(1, epochs + 1):
            toplam_kayip = 0.0

            for x, y in zip(X, Y):
                # İleri besleme: çıktıyı hesapla
                tahminler = self.ileri_besleme(x)

                # MSE kaybını biriktir (takip için)
                toplam_kayip += sum(0.5 * (t - p) ** 2
                                    for p, t in zip(tahminler, y))

                # Geri yayılım: ağırlıkları güncelle
                self.geri_yayilim(x, y)

            # Seçilen epoch'larda ilerlemeyi raporla
            if epoch <= 5 or epoch % 1000 == 0 or epoch == epochs:
                print(f"Epoch {epoch:6d} | Kayıp: {toplam_kayip:.6f}")

    # ── Tahmin ─────────────────────────────────────────────────
    def tahmin_et(self, x, esik=0.5):
        """
        Sürekli çıktıyı eşik değeriyle ikili sınıfa çevirir.
        esik=0.5 → 0.5'ten büyükse 1, küçükse 0.
        """
        prob = self.ileri_besleme(x)[0]
        return 1 if prob >= esik else 0, prob

    def degerlendir(self, X, Y):
        """Doğruluk tablosu ve başarı oranını yazdırır."""
        print("\n── XOR Sonuçları ──────────────────────────────")
        print(f"{'x1':>4} {'x2':>4} {'Beklenen':>10} {'Ham çıktı':>11} {'Tahmin':>8} {'OK?':>5}")
        print("─" * 50)
        dogru = 0
        for x, y in zip(X, Y):
            sinif, prob = self.tahmin_et(x)
            hedef = y[0]
            ok = sinif == hedef
            dogru += ok
            print(f"{x[0]:>4} {x[1]:>4} {hedef:>10} {prob:>11.4f} "
                  f"{sinif:>8} {'✓' if ok else '✗':>5}")
        print(f"\nDoğruluk: {dogru}/{len(X)} ({dogru/len(X)*100:.0f}%)")


# ─────────────────────────────────────────────────────────────────
#  Veri Seti — XOR Doğruluk Tablosu
# ─────────────────────────────────────────────────────────────────
#
#   x1  x2  →  XOR
#    0   0  →   0
#    0   1  →   1
#    1   0  →   1
#    1   1  →   0
#
#  Bu dört nokta doğrusal olarak AYRILAMAZ — tek bir düz çizgi
#  0'ları 1'lerden ayıramaz. MLP'nin gizli katmanı bu sorunu çözer.
# ─────────────────────────────────────────────────────────────────

X = [[0, 0], [0, 1], [1, 0], [1, 1]]
Y = [[0],    [1],    [1],    [0]]    # XOR çıktıları

# ─────────────────────────────────────────────────────────────────
#  Modeli Oluştur ve Eğit
# ─────────────────────────────────────────────────────────────────

random.seed(42)   # tekrar edilebilirlik için sabit tohum

model = MLP(
    n_giris      = 2,    # x1 ve x2
    n_gizli      = 2,    # gizli katmanda 2 nöron
    n_cikis      = 1,    # tek çıkış (XOR değeri)
    ogrenme_hizi = 0.5   # η: çok büyük → kararsız, çok küçük → yavaş
)

print("=" * 50)
print("  MLP EĞİTİMİ — XOR PROBLEMİ")
print("=" * 50)

model.egit(X, Y, epochs=10000)

# ─────────────────────────────────────────────────────────────────
#  Değerlendirme
# ─────────────────────────────────────────────────────────────────

model.degerlendir(X, Y)

# ─────────────────────────────────────────────────────────────────
#  Öğrenilen Ağırlıklar
# ─────────────────────────────────────────────────────────────────

print("\n── Öğrenilen Ağırlıklar ───────────────────────────")
print("Giriş → Gizli (W1):")
for i, (w, b) in enumerate(zip(model.W1, model.b1)):
    print(f"  h{i+1}: w={[round(v,3) for v in w]}, bias={b:.3f}")

print("\nGizli → Çıkış (W2):")
for i, (w, b) in enumerate(zip(model.W2, model.b2)):
    print(f"  o{i+1}: w={[round(v,3) for v in w]}, bias={b:.3f}")
