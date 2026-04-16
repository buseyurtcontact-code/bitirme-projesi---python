'''
TEK KATMANLI PERCEPTRON MODELLERİ: AND ve OR Problemleri

AND ve OR problemleri için tek katmanlı bir perceptron modeli eğitimi ve değerlendirmesi.
AND problemi:
x1 | x2 | çıktı
0  | 0  | 0
0  | 1  | 0
1  | 0  | 0
1  | 1  | 1

and mantık operatörünün doğruluk tablosu, x1 ve x2 değerlerine göre çıktıyı öğretmeye çalışıyoruz.

OR problemi:
x1 | x2 | çıktı
0  | 0  | 0
0  | 1  | 1
1  | 0  | 1
1  | 1  | 1

or mantık operatörünün doğruluk tablosu da benzer şekilde x1 ve x2 değerlerine göre çıktıyı öğretmeye çalışıyoruz.

'''
import random

# ── Yardımcı fonksiyonlar ──────────────────────────────────────────
def step(x):
    """Basamak (Heaviside) aktivasyon fonksiyonu"""
    return 1 if x >= 0 else 0 # Net girdi sıfır veya pozitifse 1, negatifse 0 döner

# ── Perceptron sınıfı ──────────────────────────────────────────────
class Perceptron:
    def __init__(self, n_inputs, lr=0.1):
        self.weights = [0.0] * n_inputs   # Başlangıçta tüm ağırlıklar sıfır
        self.bias    = 0.0                # Başlangıçta bias sıfır
        self.lr      = lr                  # Öğrenme hızı (learning rate)

    def predict(self, inputs):
        """Net girdi hesapla, basamak fonksiyonu uygula"""
        net = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias  # w·x + b formülümüz
        return step(net) # Aktivasyon fonksiyonu ile çıktı üret

    def train(self, X, y, epochs=100):
        """Perceptron öğrenme kuralı: w ← w + η·(y−ŷ)·x"""
        for epoch in range(1, epochs + 1): # Belirtilen epoch sayısı kadar döngü
            total_error = 0 # Epoch başındaki toplam hatayı sıfırdan başlattık
            for inputs, target in zip(X, y):  # Her örnek için
                pred  = self.predict(inputs) # Tahmini hesapla
                error = target - pred # Hata = gerçek - tahmin
                total_error += abs(error) # Toplam hataya ekle

                # Ağırlık güncelle
                self.weights = [w + self.lr * error * x
                                for w, x in zip(self.weights, inputs)]
                 # Bias güncelle
                self.bias   += self.lr * error

            # İlk 5 epoch veya hata sıfırsa ağırlıkları yazdır
            if epoch <= 5 or total_error == 0:
                print(f"Epoch {epoch:3d} | Hata: {total_error} | "
                      f"w={[round(w,3) for w in self.weights]}, b={self.bias:.3f}")
             # Eğer hata sıfırsa eğitim bitir
            if total_error == 0:
                print(f"  → Epoch {epoch}'de yakınsadı!\n")
                break

    def evaluate(self, X, y, gate_name):
        """Doğruluk tablosunu göster"""
        print(f"\n── {gate_name} Kapısı Sonuçları ──")
        print(f"{'x1':>4} {'x2':>4} {'Beklenen':>9} {'Tahmin':>7} {'OK?':>5}")
        print("─" * 34)
        correct = 0
        for inputs, target in zip(X, y):  # Her örnek için
            pred = self.predict(inputs)  # Tahmini hesapla
            ok   = pred == target   # Doğru mu kontrol et
            correct += ok   # Doğru sayısını artır

            print(f"{inputs[0]:>4} {inputs[1]:>4} {target:>9} {pred:>7} {'✓' if ok else '✗':>5}")
         # Sonuçları özetle
        print(f"\nDoğruluk: {correct}/{len(y)} ({correct/len(y)*100:.0f}%)")

# ── Veri setleri ───────────────────────────────────────────────────
X = [[0, 0], [0, 1], [1, 0], [1, 1]] # Girdi kombinasyonları

AND_y = [0, 0, 0, 1] # AND kapısının çıktıları
OR_y  = [0, 1, 1, 1] # OR kapısının çıktıları

# ── AND kapısı ─────────────────────────────────────────────────────
print("=" * 40)
print("   AND KAPISI EĞİTİMİ")
print("=" * 40)
p_and = Perceptron(n_inputs=2, lr=0.1)  # 2 girişli perceptron oluştur
p_and.train(X, AND_y, epochs=100)  # AND verisiyle eğit
p_and.evaluate(X, AND_y, "AND")  # Sonuçları değerlendir

# ── OR kapısı ──────────────────────────────────────────────────────
print("\n" + "=" * 40)
print("   OR KAPISI EĞİTİMİ")
print("=" * 40)
p_or = Perceptron(n_inputs=2, lr=0.1)  # 2 girişli perceptron oluştur
p_or.train(X, OR_y, epochs=100)  # OR verisiyle eğit
p_or.evaluate(X, OR_y, "OR")  # Sonuçları değerlendir
