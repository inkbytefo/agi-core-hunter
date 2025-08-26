# AGI Çekirdek Avcısı - Kapsamlı Yol Haritası

**Proje Başlangıcı:** 26 Ağustos 2025  
**Son Güncelleme:** 26 Ağustos 2025  
**Vizyon:** Yapay Genel Zekâ'nın temel ilkelerini sistematik olarak keşfetmek

---

## 🎯 Proje Hedefleri

### Ana Hedef
Zekânın altında yatan temel, birleştirici ilkeleri bulmak ve test etmek

### Alt Hedefler
1. **Bilimsel:** Falsifikasyon odaklı AGI araştırma metodolojisi geliştirmek
2. **Teknik:** Açık kaynak AGI ilke karşılaştırma platformu oluşturmak
3. **Akademik:** Peer-reviewed makaleler ve konferans sunumları
4. **Toplumsal:** AGI araştırma topluluğuna katkıda bulunmak

---

## 📋 Faz 1: Temel Altyapı (TAMAMLANDI ✅)

### 1.1 Proje Kurulumu ✅
- [x] Monorepo yapısı oluşturma
- [x] Modüler kod organizasyonu
- [x] Açık kaynak hazırlığı (setup.py, requirements.txt, LICENSE)
- [x] Git repository kurulumu
- [x] README ve temel dokümantasyon

### 1.2 Temel Modüller ✅
- [x] `BaseAgent` soyut sınıfı
- [x] `GridWorld` test ortamı
- [x] Metrik hesaplama araçları (`MetricsTracker`)
- [x] Test altyapısı (`test_setup.py`)
- [x] Yardımcı fonksiyonlar

### 1.3 İlk İlke: MDL (Minimum Description Length) ✅
- [x] `MDLAgent` implementasyonu (β-VAE tabanlı)
- [x] Teori kartı (`MDL_PRINCIPLE.md`)
- [x] İlk deney tasarımı (`01_mdl_vs_ood`)
- [x] Manifest sistemi (yeniden üretilebilirlik)

### 1.4 Dokümantasyon ✅
- [x] Proje tanıtım belgesi (`PROJE_TANITIM.md`)
- [x] Teknik mimari belgesi (`PROJE_TEKNIKMIMARI.md`)
- [x] Proje durumu takibi (`PROJECT_STATUS.md`)
- [x] Bu yol haritası (`ROADMAP.md`)

**Faz 1 Tamamlanma:** ✅ %100 (26 Ağustos 2025)

---

## 🧪 Faz 2: İlk Deney ve Analiz (DEVAM EDİYOR 🔄)

### 2.1 MDL vs OOD Deneyi 🔄
- [x] Deney kurulumu ve başlatma
- [x] Wandb entegrasyonu
- [x] 3 farklı β değeri testi (0.1, 1.0, 5.0)
- [ ] **DEVAM EDİYOR:** 5000 episode eğitim (her agent için)
- [ ] OOD senaryoları testi
- [ ] Sonuç analizi (`eval.ipynb`)

### 2.2 İlk Sonuç Değerlendirmesi 📅
- [ ] Hipotez testi (β ↑ → OOD performansı ↑ ?)
- [ ] İstatistiksel analiz
- [ ] Görselleştirmeler
- [ ] MDL teori kartı güncelleme
- [ ] İlk bulgular raporu

### 2.3 Metodoloji Doğrulama 📅
- [ ] Deney yeniden üretilebilirlik testi
- [ ] Metrik güvenilirlik analizi
- [ ] Minimal ortam etkinlik değerlendirmesi

**Faz 2 Hedef Tamamlanma:** 28 Ağustos 2025

---

## 🔬 Faz 3: MDL İlkesi Derinleştirme (PLANLI 📅)

### 3.1 MDL Varyasyonları 📅
- [ ] **Deney 02:** Farklı latent boyutları (2, 4, 8, 16, 32)
  - [ ] Sanity check: 100 episode pilot test
  - [ ] Full run: 2000 episode (GPU optimizasyonu)
- [ ] **Deney 03:** Farklı VAE mimarileri (β-VAE, β-TCVAE, Factor-VAE)
- [ ] **Deney 04:** Baseline karşılaştırmaları (PCA, ICA, Random)
- [ ] **Deney 05:** Farklı β programları (adaptive, scheduled)

### 3.2 Ortam Çeşitliliği 📅
- [ ] Daha büyük grid boyutları (16x16, 32x32)
- [ ] Dinamik engeller
- [ ] Çoklu hedef senaryoları
- [ ] **Köprü Çıktısı:** Faz 4 için standardize OOD ortam seti

### 3.3 Gelişmiş OOD Testleri 📅
- [ ] Görsel değişiklikler (renk, doku)
- [ ] Fizik kuralı değişiklikleri
- [ ] Ödül yapısı değişiklikleri
- [ ] **Köprü Çıktısı:** Nedensel müdahale test ortamları

### 3.4 Mini-APCB Lansmanı 📅
- [ ] MDL + Baseline karşılaştırma platformu
- [ ] Topluluk beta testi
- [ ] İlk external submission'lar

**Faz 3 Hedef Tamamlanma:** 15 Eylül 2025

---

## 🧠 Faz 4: İkinci İlke - Nedensellik (PLANLI 📅)

### 4.1 Nedensel Agent Geliştirme 📅
- [ ] `CausalAgent` implementasyonu
- [ ] Yapısal Nedensel Model (SCM) entegrasyonu
- [ ] Do-calculus planlama algoritması
- [ ] Nedensel keşif modülü

### 4.2 Nedensellik Teori Kartı 📅
- [ ] `CAUSALITY_PRINCIPLE.md` oluşturma
- [ ] Test edilebilir hipotezler formülasyonu
- [ ] Minimal test ortamı tasarımı
- [ ] Başarı metrikleri tanımlama

### 4.3 Nedensellik Deneyleri 📅
- [ ] **Deney 06:** Nedensel vs korelasyonel öğrenme
- [ ] **Deney 07:** Müdahale adaptasyonu
- [ ] **Deney 08:** Nedensel transfer öğrenme
- [ ] **Deney 09:** Confounding robustluğu

### 4.4 Nedensel Ortamlar 📅
- [ ] Nedensel graf tabanlı ortamlar
- [ ] Müdahale simülasyonu
- [ ] Confounding değişkenler
- [ ] Temporal nedensellik

**Faz 4 Hedef Tamamlanma:** 15 Ekim 2025

---

## ⚡ Faz 5: Üçüncü İlke - Serbest Enerji (PLANLI 📅)

### 5.1 Serbest Enerji Agent'ı 📅
- [ ] `FEPAgent` implementasyonu (Active Inference)
- [ ] Varyasyonel serbest enerji minimizasyonu
- [ ] Üretken model öğrenme
- [ ] Aktif algı ve eylem seçimi

### 5.2 Serbest Enerji Teori Kartı 📅
- [ ] `FREE_ENERGY_PRINCIPLE.md` oluşturma
- [ ] Belirsizlik azaltma hipotezleri
- [ ] Minimal test senaryoları
- [ ] Sürpriz ve entropi metrikleri

### 5.3 Serbest Enerji Deneyleri 📅
- [ ] **Deney 10:** Belirsizlik azaltma davranışı
- [ ] **Deney 11:** Aktif öğrenme vs pasif öğrenme
- [ ] **Deney 12:** Çevresel model doğruluğu
- [ ] **Deney 13:** Adaptif keşif stratejileri

**Faz 5 Hedef Tamamlanma:** 15 Kasım 2025

---

## 🏆 Faz 6: İlkeler Arası Karşılaştırma (PLANLI 📅)

### 6.1 Kapsamlı Benchmark 📅
- [ ] **Deney 14:** Tüm ilkelerin aynı ortamda testi
- [ ] Çoklu metrik karşılaştırması
- [ ] İstatistiksel anlamlılık testleri
- [ ] Performans profili analizi

### 6.2 APCB (AGI Principle Comparison Benchmark) 📅
- [ ] Standardize edilmiş test süiti
- [ ] Otomatik değerlendirme sistemi
- [ ] Leaderboard implementasyonu
- [ ] Topluluk submission sistemi

### 6.3 Meta-Analiz 📅
- [ ] İlkeler arası korelasyon analizi
- [ ] Ortam türü vs ilke etkinliği
- [ ] Hibrit yaklaşım potansiyeli
- [ ] Teorik birleştirme imkanları

**Faz 6 Hedef Tamamlanma:** 15 Aralık 2025

---

## 📚 Faz 7: Akademik Çıktılar (PLANLI 📅)

### 7.1 Akademik Hızlanma Stratejisi 📅

#### Workshop Paper (Hızlı Giriş)
**Hedef:** ICLR 2026 Workshop  
**Deadline:** 15 Aralık 2025  
**Başlık:** "Falsifiable AGI Principles: Early Results from Minimal Laboratories"
- [ ] 4 sayfa metodoloji + MDL sonuçları
- [ ] Topluluk feedback toplama
- [ ] Ağ kurma fırsatı

#### Ana Makale (Kapsamlı)
**Hedef Dergi:** ICLR 2026 (Ana Track)  
**Deadline:** 1 Ekim 2025 → **REVİZE:** 1 Şubat 2026  
**Başlık:** "Systematic Falsification of AGI Principles: A Minimal Laboratory Approach"
- [ ] Literatür taraması
- [ ] 3 ilke karşılaştırması (MDL + Nedensellik + FEP)
- [ ] APCB benchmark tanıtımı
- [ ] Kapsamlı sonuç analizi

### 7.2 İkinci Makale: Benchmark 📅
**Hedef Dergi:** NeurIPS 2026  
**Deadline:** 15 Mayıs 2026  
**Başlık:** "APCB: An Open Benchmark for Comparing AGI Principles"

- [ ] Benchmark detaylandırması
- [ ] Çoklu ilke sonuçları
- [ ] Topluluk kullanım analizi
- [ ] Makale yazımı

### 7.3 Konferans Sunumları 📅
- [ ] **ICLR 2026:** Metodoloji sunumu
- [ ] **NeurIPS 2026:** Benchmark sunumu
- [ ] **ICML 2026:** Workshop organizasyonu
- [ ] Yerel konferanslar ve seminerler

**Faz 7 Hedef Tamamlanma:** Haziran 2026

---

## 🌍 Faz 8: Topluluk ve Yaygınlaştırma (PLANLI 📅)

### 8.1 Açık Kaynak Geliştirme 📅
- [ ] GitHub star hedefi: 1000+
- [ ] Katkıda bulunma kılavuzu
- [ ] Issue ve PR yönetimi
- [ ] Topluluk moderasyonu

### 8.2 Eğitim İçerikleri 📅
- [ ] **Deney Günlüğü Serisi** (Medium/Substack)
  - [ ] "MDL İlkesi: İlk Sonuçlar" (Bu hafta)
  - [ ] "Nedensellik vs Korelasyon: Hangi Agent Kazanır?" 
  - [ ] "AGI İlkelerini Nasıl Test Ediyoruz?"
- [ ] YouTube video serisi (10+ video)
- [ ] Online kurs materyali
- [ ] Interaktif demo'lar (Streamlit/Gradio)

### 8.3 Endüstri İşbirlikleri 📅
- [ ] Tech şirketleri ile partnership
- [ ] Araştırma laboratuvarları işbirliği
- [ ] Startup inkübatör programları
- [ ] Danışmanlık hizmetleri

**Faz 8 Hedef Tamamlanma:** Aralık 2026

---

## 🔮 Faz 9: Gelecek Vizyonu (UZUN VADELİ 📅)

### 9.1 Gelişmiş İlkeler 📅
- [ ] Dördüncü İlke: Meta-öğrenme
- [ ] Beşinci İlke: Öz-organizasyon
- [ ] Altıncı İlke: Emergent complexity
- [ ] İlke hibridizasyonu

### 9.2 Gerçek Dünya Uygulamaları 📅
- [ ] Robotik sistemlerde test
- [ ] Doğal dil işleme uygulamaları
- [ ] Oyun AI'ında implementasyon
- [ ] Otonom sistemlerde kullanım

### 9.3 Teorik Birleştirme 📅
- [ ] Unified Theory of Intelligence
- [ ] Matematiksel formalizasyon
- [ ] Fiziksel temeller
- [ ] Bilişsel bilim bağlantıları

**Faz 9 Hedef Tamamlanma:** 2027-2030

---

## 📊 İlerleme Takibi

### Tamamlanan Fazlar
- ✅ **Faz 1:** Temel Altyapı (100%)

### Devam Eden Fazlar  
- 🔄 **Faz 2:** İlk Deney ve Analiz (60%)

### Yaklaşan Fazlar
- 📅 **Faz 3:** MDL Derinleştirme (0%)
- 📅 **Faz 4:** Nedensellik İlkesi (0%)

### Genel İlerleme
**Toplam Proje İlerlemesi:** 15% (1.5/10 faz)

---

## 🎯 Kritik Başarı Faktörleri

### Teknik
- [ ] Yeniden üretilebilir deneyler
- [ ] Sağlam istatistiksel analiz
- [ ] Ölçeklenebilir kod mimarisi
- [ ] Kapsamlı test coverage

### Akademik
- [ ] Peer-reviewed yayınlar
- [ ] Konferans kabulleri
- [ ] Atıf sayısı artışı
- [ ] Akademik ağ genişlemesi

### Topluluk
- [ ] GitHub star sayısı
- [ ] Aktif katkıda bulunanlar
- [ ] Endüstri adaptasyonu
- [ ] Medya görünürlüğü

### Etki
- [ ] AGI araştırma metodolojisine katkı
- [ ] Yeni araştırma yönelimlerinin tetiklenmesi
- [ ] Pratik uygulamaların geliştirilmesi
- [ ] Bilimsel paradigma değişimine katkı

---

## 🤖 Otomasyon ve Verimlilik Stratejileri

### Deney Fabrikası Otomasyonu
- [ ] **CI/CD Pipeline:** Her commit'te otomatik test
- [ ] **Otomatik Rapor:** Her deney sonrası `eval_report.md` güncellemesi
- [ ] **Sanity Check Sistemi:** Full deney öncesi 100-episode pilot
- [ ] **GPU Optimizasyonu:** Batch processing ve paralel eğitim

### Manifest-Driven Development
- [ ] **Deney Şablonları:** Yeni ilke için otomatik scaffold
- [ ] **Parametre Sweep:** Grid search otomasyonu
- [ ] **Sonuç Karşılaştırma:** Otomatik A/B test raporları
- [ ] **Versiyon Kontrolü:** Deney sonuçları için Git LFS

### Topluluk Katkı Otomasyonu
- [ ] **Issue Templates:** Yeni ilke önerisi şablonları
- [ ] **PR Checklist:** Kod kalitesi ve deney standardları
- [ ] **Otomatik Benchmark:** External submission'lar için test suite
- [ ] **Leaderboard Bot:** Sonuçları otomatik güncelleme

---

## 🚨 Risk Faktörleri ve Azaltma Stratejileri

### Teknik Riskler
- **Risk:** Deney sonuçları belirsiz çıkabilir
- **Azaltma:** Çoklu hipotez testi, robust istatistik

- **Risk:** Kod karmaşıklığı artabilir  
- **Azaltma:** Modüler tasarım, sürekli refactoring

### Akademik Riskler
- **Risk:** Makale reddedilme
- **Azaltma:** Erken feedback, çoklu dergi hedefi

- **Risk:** Rekabet artışı
- **Azaltma:** Hızlı iterasyon, açık kaynak avantajı

### Kaynak Riskleri
- **Risk:** Zaman kısıtları
- **Azaltma:** Öncelik sıralaması, MVP yaklaşımı

- **Risk:** Hesaplama kaynağı yetersizliği
- **Azaltma:** Bulut servisleri, akademik hibeler

---

## 📞 İletişim ve Koordinasyon

### Haftalık Değerlendirme
- Her Pazartesi: İlerleme raporu
- Her Çarşamba: Teknik review
- Her Cuma: Strateji planlaması

### Aylık Milestone'lar
- Ay sonu: Faz tamamlanma değerlendirmesi
- Sonraki ay planlaması
- Risk değerlendirmesi güncelleme

### Üç Aylık Stratejik Review
- Genel yön değerlendirmesi
- Hedef revizyon ihtiyacı
- Kaynak yeniden tahsisi

---

**Son Güncelleme:** 26 Ağustos 2025  
**Sonraki Review:** 26 Eylül 2025  
**Proje Durumu:** 🟢 Aktif ve Planında