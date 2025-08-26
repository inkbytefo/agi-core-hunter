# Teori Kartı: Minimum Description Length (MDL) İlkesi

**Tarih:** 26.08.2025  
**Durum:** Aktif Test Aşamasında  
**İlgili Deneyler:** `01_mdl_vs_ood`

---

## 🎯 Temel İddia

**"Zekâ, veriyi en kısa şekilde açıklama (sıkıştırma) yeteneğidir."**

Bir model ne kadar verimli sıkıştırma yaparsa, o kadar iyi genelleme yapar. Bu, Occam'ın Usturası'nın matematiksel formülasyonudur.

---

## 📐 Matematiksel Formülasyon

### Kolmogorov Karmaşıklığı
```
K(x) = min{|p| : U(p) = x}
```
- `K(x)`: x dizisinin Kolmogorov karmaşıklığı
- `|p|`: program p'nin uzunluğu
- `U(p)`: evrensel Turing makinesi U'nun p programını çalıştırması

### Pratik MDL (İki Parçalı Kod)
```
L(H, D) = L(H) + L(D|H)
```
- `L(H)`: Hipotez H'nin açıklama uzunluğu
- `L(D|H)`: H verildiğinde veri D'nin açıklama uzunluğu
- **Hedef:** `L(H, D)`'yi minimize et

---

## 🔬 Test Edilebilir Öngörüler

1. **OOD Robustluğu:** MDL düzenlemesi yapılan modeller, dağıtım dışı senaryolarda daha az performans kaybı yaşar
2. **Örnek Verimliliği:** Daha az veriyle daha iyi genelleme yapar
3. **Transfer Öğrenme:** Sıkıştırılmış temsiller, yeni görevlere daha kolay transfer olur
4. **Gürültü Dayanıklılığı:** Sıkıştırma, doğal bir gürültü filtreleme etkisi yapar

---

## 🧪 Minimal Test Ortamı

### Grid World Navigation
- **Ortam:** 10x10 grid, rastgele engeller
- **Görev:** Başlangıçtan hedefe en kısa yol bulma
- **MDL Uygulaması:** Durum temsilini VAE ile sıkıştır
- **Test:** Eğitimde görülmeyen grid konfigürasyonlarında performans

### Kontrol Grubu
- Standart DQN (sıkıştırma yok)
- β-VAE DQN (farklı β değerleri)

---

## 📊 Başarı Metrikleri

1. **OOD Genelleme Skoru:** `(Perf_OOD / Perf_Train) * 100`
2. **Sıkıştırma Oranı:** `Original_Dim / Latent_Dim`
3. **Örnek Verimliliği:** Hedef performansa ulaşmak için gereken episode sayısı
4. **Transfer Hızı:** Yeni göreve adaptasyon için gereken adım sayısı

---

## 🚨 Potansiel Çürütme Senaryoları

- Aşırı sıkıştırma performansı düşürürse
- OOD'da sıkıştırılmamış model daha iyi performans gösterirse  
- Transfer öğrenmede MDL avantaj sağlamazsa
- Hesaplama maliyeti faydayı aşarsa

---

## 📚 Temel Kaynaklar

- Rissanen, J. (1978). "Modeling by shortest data description"
- Grünwald, P. (2007). "The Minimum Description Length Principle"
- Schmidhuber, J. (2010). "Formal Theory of Creativity, Fun, and Intrinsic Motivation"

---

## 🔄 Güncellemeler

- **26.08.2025:** İlk kart oluşturuldu
- *Deney sonuçlarına göre güncellenecek...*