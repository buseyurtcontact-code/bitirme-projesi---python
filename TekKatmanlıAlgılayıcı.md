1-) Yardımcı Fonksiyon
step(x): Heaviside (basamak) aktivasyon fonksiyonu.
Eğer net girdi ≥ 0 ise çıktı 1, aksi halde 0.

2-) Perceptron Sınıfı
__init__ : Başlangıçta tüm ağırlıkları ve biası sıfır yapıyor. Öğrenme hızı (lr) varsayılan olarak 0.1

predict: Girdi ile ağırlıkların çarpımını ve bias’ı topluyor, sonra step fonksiyonuyla çıktı veriyor.

train: Eğitim verilerini (X, y) kullanarak perceptron öğrenme kuralını uygular:
w <- w + η*(y-ŷ)*x
b <- b + η*(y-ŷ)

Epoch başına toplam hatayı hesaplar. Eğer hata sıfır olursa eğitim durur (yakınsama).
evaluate: Eğitim sonrası doğruluk tablosunu yazdırır. Beklenen çıktı ile tahmini karşılaştırır.

3-) Veri Setleri
X = [[0,0], [0,1], [1,0], [1,1]] → iki girişli mantık kapılarının tüm kombinasyonları.
AND_y = [0,0,0,1] → AND kapısının doğruluk tablosu.
OR_y = [0,1,1,1] → OR kapısının doğruluk tablosu.

4-) Eğitim ve Test
AND Kapısı Eğitimi: Perceptron, AND fonksiyonunu öğreniyor. Sonra doğruluk tablosu yazdırılıyor.
OR Kapısı Eğitimi: Aynı şekilde OR fonksiyonunu öğreniyor ve test ediliyor.

Özetle: Bu kod, perceptronun mantık kapılarını (AND ve OR) öğrenebildiğini gösteren bir eğitim ve test simülasyonu yapıyor. Yani yapay sinir ağlarının en temel örneklerinden birini uygulamalı olarak gösteriyor.


```mermaid
flowchart TD
    A([Başlangıç]) --> B[Weights = 0, Bias = 0, Learning Rate belirle]
    B --> C[Eğitim Döngüsü Başlat]
    C --> D[Tahmin Hesapla: net = Σw·x + b]
    D --> E[Step Fonksiyonu Uygula: ŷ = 1 if net ≥ 0 else 0]
    E --> F[Hata Hesapla: error = y - ŷ]
    F --> G[Ağırlıkları Güncelle: w ← w + LearninRate·error·x]
    G --> H[Bias Güncelle: b ← b + LearninRate·error]
    H --> I[Toplam Hata Hesapla]
    I --> J{Toplam Hata = 0?}
    J --> Evet --> K[Yakınsama! Eğitim Bitir]
    K --> Hayır --> C
    L --> B[Değerlendirme: Doğruluk Tablosu Yazdır]

    #düzenlenecek 
