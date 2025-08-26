# Proje: AGI Çekirdek Avcısı (AGI Core Hunter)

**Sürüm:** 1.0
**Tarih:** 26.08.2025
**Sorumlular:** Inkbytefo

---

## 1. Özet

Bu proje, Yapay Genel Zekâ'nın (AGI) altında yatan temel, birleştirici ilkeleri bulmak için tasarlanmış, hipotez-odaklı ve deneysel bir araştırma programıdır. Mevcut "daha büyük model = daha iyi zekâ" paradigmasının ötesine geçerek, zekânın temel "fiziğini" anlamayı hedefliyoruz. Metodolojimiz, farklı teorik adayları (Algoritmik Olasılık, Nedensellik, Serbest Enerji İlkesi vb.) basitleştirilmiş simülasyon ortamlarında, falsifikasyon odaklı deneylerle sistematik olarak test etmektir. Projenin nihai çıktısı, yalnızca başarılı modeller değil, aynı zamanda bu ilkeleri karşılaştırmak için tasarlanmış açık kaynaklı bir benchmark ve aydınlatıcı başarısızlıkları belgeleyen bir "ilke kütüphanesi" olacaktır.

---

## 2. Problem Tanımı ve Motivasyon

Günümüz yapay zekâsı, etkileyici yeteneklerine rağmen temel bir sorundan muzdariptir: **Anlayıştan yoksun genelleme.** Büyük dil ve görüntü modelleri, devasa veri setlerindeki istatistiksel korelasyonları ezberlemekte ustadır, ancak bu onları dağıtım dışı (OOD) senaryolarda kırılgan, nedensel akıl yürütmede zayıf ve öngörülemez kılar. AGI'ye ulaşmak, sadece mevcut mimarileri ölçeklendirmekten değil, zekânın kendisini yöneten temel prensipleri keşfetmekten geçer.

Bu proje, bu temel prensibi bulma arayışıdır.

---

## 3. Pusula: Çalışma Varsayımımız

Zekânın tek bir ilkeye indirgenebileceği varsayımıyla yola çıkıyoruz. Bu ilkenin şu üç temel alanda kesiştiğini öne sürüyoruz:

1. **Bilgi-Enerji-Nedensellik Üçgeni:** Zekâ, fiziksel kısıtlar (enerji) altında, dünyanın nedensel yapısını kullanarak bilgiyi verimli bir şekilde işleyen bir süreçtir.

2. **İç Dünya Modeli:** Zeki ajanlar, dünyayı pasif bir şekilde gözlemlemez; geleceği öngörmek, eylemlerinin sonuçlarını simüle etmek ve plan yapmak için aktif olarak bir iç model oluşturur ve kullanır.

3. **Sıkıştırma ve Örüntü Ekonomisi:** Genelleme, veriyi en öz şekilde açıklama (sıkıştırma) yeteneğinden doğar. Zekâ, en az karmaşıklıkla en fazla öngörü gücünü elde etme sanatıdır.

**Hedefimiz:** Bu üçgeni tek bir matematiksel çekirdekte birleştiren aday ilkeleri bulmak, test etmek ve elemek.

---

## 4. Araştırma Metodolojisi: Hızlı Falsifikasyon Döngüsü

Büyük teorileri yıllarca tartışmak yerine, onları hızla test edip eleyen bir döngü izleyeceğiz:

**Hipotez → Deney → Değerlendirme**

- **Hipotez:** Her aday ilkeyi, test edilebilir, tek cümlelik bir iddiaya indirgemek. (Örn: "MDL ile düzenlenen temsiller OOD genellemede daha iyidir.")
- **Deney:** Hipotezi test etmek için tasarlanmış **minimal canlı laboratuvarlar** kurmak. Bunlar, karmaşık SOTA benchmark'ları yerine, ilkenin öngörüsünü izole edebilen basit oyuncak dünyalar (grid-world, 2D fizik) olacaktır.
- **Değerlendirme:** Başarıyı, SOTA skorlarıyla değil, ilkeye özgü metriklerle ölçmek: OOD robustluğu, örnek verimliliği, nedensel müdahaleye adaptasyon, hesaplama maliyeti. Bir ilkenin öngörüsü deneyde çürütülürse, hipotez terk edilir veya revize edilir.

---

## 5. Ana Araştırma Hatları (Aday İlkeler)

Aşağıdaki teorik çerçeveleri sistematik olarak inceleyeceğiz:

- **A. Algoritmik Olasılık & Evrensel Ajanlar:** Zekâ, en kısa açıklamayı (Kolmogorov Karmaşıklığı/MDL) bulma ve kullanma yeteneğidir. (Solomonoff, Hutter)
- **B. Nedensellik & Temsil:** Zekâ, korelasyonları değil, dünyanın altında yatan neden-sonuç mekanizmalarını modelleme ve bunlara müdahale etme yeteneğidir. (Pearl, Schölkopf)
- **C. Serbest Enerji İlkesi & Etkin Öngörü:** Zekâ, bir ajanın dünya modeli ile duyusal girdileri arasındaki sürprizi (serbest enerji) en aza indirme sürecidir. Algı ve eylem bu amaç için birleşir. (Friston)
- **D. Dünya Modelleri & Öz-denetimli Öğrenme:** Zekâ, dünyanın hızlı bir simülasyonunu (dünya modeli) öğrenip eylemlerini bu "hayal" içinde planlama yeteneğidir. (Schmidhuber, LeCun)
- **E. Hesaplamanın Termodinamiği:** Zekânın fiziksel alt sınırlarını ve maliyetini anlamak. (Landauer)
- **F. Öz-iyileşen Sistemler:** Zekânın, kendi kodunu kanıta dayalı olarak değiştirebilen ve geliştirebilen sistemlerden doğacağı fikri. (Schmidhuber)

---

## 6. Beklenen Çıktılar

1. **Deneysel Sonuç Raporları:** Her aday ilke için yapılan deneylerin, başarıların ve özellikle aydınlatıcı başarısızlıkların yayınlanması.
2. **"AGI İlke Karşılaştırma Benchmark'ı" (APCB):** Aday ilkelerin OOD genelleme, nedensel transfer ve verimlilik gibi eksenlerde karşılaştırılabileceği, açık kaynaklı bir test ortamı ve metrik seti.
3. **Açık Kaynak Kod Deposu:** Tüm ajanların, ortamların ve analiz araçlarının yeniden üretilebilir bir şekilde paylaşıldığı bir `agi_core_hunter` reposu.
4. **"İlke Kartları" Kütüphanesi:** Her ilkeyi, temel formülünü, aykırı öngörüsünü ve minimal deneyini özetleyen tek sayfalık belgeler.

---

## 7. Proje Felsefesi (Neyi Yapmayacağız)

- **Sadece SOTA Skoru Kovalamak:** Amacımız liderlik tablolarında birinci olmak değil, temel ilkeleri anlamaktır.
- **"Büyük Model = Çözüm" Varsayımı:** Parametre sayısını artırmayı, bir ilkenin testini bulandıran bir değişken olarak görüyoruz.
- **Kör Optimizasyon:** Bir hipotez çalışmadığında, onu değiştirmeden sadece hiperparametreleri veya veri miktarını artırmayacağız.
- **Tek Demo Yanılgısı:** Etkileyici bir demo, bir teorinin kanıtı değildir. Sistematik ve tekrarlanabilir sonuçlar esastır.