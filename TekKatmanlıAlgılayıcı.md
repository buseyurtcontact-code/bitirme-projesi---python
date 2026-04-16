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
