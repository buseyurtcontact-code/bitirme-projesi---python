XOR için Çok Katmanlı Algılayıcı Algoritması
1-) Başlatma (Initialization)

Giriş katmanı: 
𝑥1,𝑥2

Gizli katman: birkaç nöron (örneğin 2 nöron)

Çıkış katmanı: 1 nöron (XOR çıktısı)

Ağırlıkları ve bias değerlerini küçük rastgele sayılarla başlat.

Öğrenme hızını (𝜂 yani learning rate) belirle.

2-) İleri Yayılım (Forward Propagation)

Her gizli nöron için:
   hj = f(∑(wij*xi)+bj)
   #burada f genellikle sigmoid, tanh veya ReLU aktivasyon fonksiyonudur.
   
Çıkış nöronu için:
   ŷ = f(∑(wj*hj)+b)

3-) Hata Hesaplama (Error Calculation)
Gerçek çıktı 𝑦 ile tahmin ŷ arasındaki farkı hesapla.
Genellikle Mean Squared Error (MSE) veya Cross-Entropy Loss kullanılır:
E = 1/2*(𝑦-ŷ)^2

4-) Geri Yayılım (Backpropagation)
Çıkış katmanındaki hata türevini hesapla:
𝛿out = (𝑦-ŷ)*f'(netout)
δhidden = 𝛿out*wout*f'(nethidden)

5-) Ağırlık Güncelleme (Weight Update)
Çıkış katmanı ağırlıkları:
wout <- wout + 𝜂*𝛿out*h

Gizli katman ağırlıkları:
whidden  <- whidden + 𝜂*𝛿hidden*x

bias değerleri de aynı şekilde güncellenir 

6-) Tekrar (Iteration)
Tüm eğitim örnekleri için ileri yayılım → hata → geri yayılım → güncelleme adımlarını uygula.
Epoch’lar boyunca hata azalana kadar devam et.

Değerlendirme (Evaluation)

Eğitim tamamlandıktan sonra XOR doğruluk tablosunu test et:

(0,0) → 0

(0,1) → 1

(1,0) → 1

(1,1) → 0

Özet: XOR problemi doğrusal ayrılabilir değildir, bu yüzden tek katmanlı perceptron öğrenemez. 
Çok katmanlı algılayıcı, gizli katman sayesinde doğrusal olmayan karar sınırları oluşturur ve XOR’u başarıyla öğrenir.


Açıklama
Bu diyagramda her kutunun yanındaki yorum satırları (%%) sana adımın ne işe yaradığını anlatıyor.

GitHub README dosyana bu kodu koyarsan, Mermaid desteği olan ortamlarda akış diyagramı olarak görünecek.

Akış: Başlat → İleri Yayılım → Hata → Geri Yayılım → Güncelleme → Kontrol → Yakınsama → Değerlendirme şeklinde ilerliyor.



```mermaid
flowchart TD

A[Başlatma] --> B[Ağırlıklar ve bias değerlerini küçük rastgele sayılarla başlat]
B --> C[İleri Yayılım Forward Propagation]

C --> D[Gizli Katman: net_hidden = Σw·x + b]
D --> E[Aktivasyon Fonksiyonu sigmoid/tanh/ReLU]
E --> F[Çıkış Katmanı: net_out = Σw·h + b]
F --> G[Çıkış Aktivasyonu: ŷ = fnet_out]

G --> H[Hata Hesaplama: error = y - ŷ]
H --> I[Geri Yayılım Backpropagation]

I --> J[Çıkış Katmanı Hata: δ_out = y - ŷ   ·f'net_out]
J --> K[Gizli Katman Hata: δ_hidden = δ_out·w_out·f' net_hidden]

K --> L[Ağırlık Güncelleme]
L --> M[w_out ← w_out + η·δ_out·h]
M --> N[w_hidden ← w_hidden + η·δ_hidden·x]
N --> O[Bias değerlerini güncelle]

O --> P{Toplam Hata Küçük mü?}
P -- Hayır --> C
P -- Evet --> Q[Yakınsama! Eğitim Bitir]

Q --> R[Değerlendirme: XOR doğruluk tablosunu test et]

