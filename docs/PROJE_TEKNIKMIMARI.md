# AGI Çekirdek Avcısı: Teknik Mimari ve Uygulama Planı

**Sürüm:** 1.0
**Tarih:** 24.10.2023

---

## 1. Genel Bakış

Bu belge, "AGI Çekirdek Avcısı" projesinin pratik uygulama detaylarını, teknoloji yığınını, kod yapısını ve deney akışını tanımlar. `PROJE_TANITIM.md`'de belirtilen araştırma vizyonunu hayata geçirmek için kullanılacak teknik yol haritasıdır.

---

## 2. Teknoloji Yığını ve Araç Kutusu

| Kategori              | Araç/Teknoloji                                 | Gerekçe                                                               |
| --------------------- | ---------------------------------------------- | --------------------------------------------------------------------- |
| **Programlama Dili**  | Python 3.9+                                    | Standart ekosistem, geniş kütüphane desteği.                          |
| **Derin Öğrenme**     | JAX / Flax (öncelikli), PyTorch (alternatif)   | Fonksiyonel programlama, `vmap`/`pmap` ile kolay paralelleştirme, dünya modeli/planlama için ideal. |
| **Simülasyon/Fizik**  | Brax, `gymnax` veya özel 2D motor (`pymunk`)     | JAX tabanlı, hızlı ve paralelleştirilebilir ortamlar.                   |
| **Nedensel Kütüphane** | `DoWhy`, `CausalNex`, `EconML`                 | Nedensel graf keşfi ve do-hesaplaması için standart araçlar.          |
| **Deney Yönetimi**    | Weights & Biases (WandB) / MLflow              | Metriklerin, parametrelerin ve çıktıların takibi ve karşılaştırılması. |
| **Not Alma/Belgeleme**| Obsidian, Markdown                             | Merkezi bilgi yönetimi, "İlke Kartları" ve deney notları için.        |

---

## 3. Proje Yapısı (Monorepo)

Tüm kodlar, tek ve organize bir depoda tutulacaktır.

```bash
/agi_core_hunter
├── 📜 README.md                 # Projenin ana tanıtım dosyası
├── 📚 literature/               # İlgili makaleler ve özetler
├── 📝 docs/
│   ├── proje_tanitim.md
│   ├── proje_teknikmimari.md
│   └── theory_cards/           # Her ilke için "Tek Sayfa" kartları
├── 🔬 experiments/               # Her bir yayınlanabilir deney için klasör
│   ├── 01_mdl_vs_ood/
│   │   ├── manifest.json         # Deneyin yapılandırma dosyası (hipotez, params)
│   │   ├── env.py                # Deneye özel ortam
│   │   ├── agent.py              # Deneydeki ajan(lar)
│   │   ├── train.py              # Eğitim betiği
│   │   └── eval.ipynb            # Sonuç analizi ve görselleştirme
│   └── ...
├── 🧠 src/                       # Paylaşılan kodlar
│   ├── agents/                 # Ajanların temel sınıfları (MDL, Causal, FEP...)
│   ├── envs/                   # Temel ortam sınıfları (ToyPhysics, CausalGraphGen)
│   ├── core/                   # Ortak modüller (planlama, temsil öğrenme vb.)
│   └── utils/                  # Yardımcı fonksiyonlar (logger, metrikler)
├── requirements.txt              # Proje bağımlılıkları
└── setup.py                      # Projeyi kurulabilir paket haline getirmek için
```

---

## 4. Deney Akışı ve Yönetimi

Her deney aşağıdaki standart akışı takip edecektir:

1. **Hipotez Formülasyonu:** `docs/theory_cards/` altında ilgili ilke için bir kart oluşturulur.
2. **Manifest Dosyası (`manifest.json`):** Her deney, parametrelerini ve yapılandırmasını içeren bir JSON dosyası ile başlar. Bu, %100 yeniden üretilebilirlik sağlar.
3. **Eğitim (`train.py`):** Bu betik, manifest dosyasını okur, ilgili ajanı ve ortamı `src`'den yükler, WandB'yi başlatır ve eğitimi yürütür. Tüm metrikler, hiperparametreler ve model checkpoint'leri WandB'ye loglanır.
4. **Değerlendirme (`eval.ipynb`):** Eğitim tamamlandıktan sonra, bu not defteri WandB API'si ile sonuçları çeker. Başarı metrikleri (OOD skoru, transfer başarısı vb.) hesaplanır, hipotezin doğrulanıp doğrulanmadığına dair bir "karar" verilir ve grafikler oluşturulur.

---

## 5. Ajan Modül Taslakları (Uygulama Detayları)

### `MDLAgent`:
- **Mimari:** VAE (Variational Autoencoder) tabanlı temsil öğrenme modülü. Girdi (gözlem) `z` latent vektörüne kodlanır.
- **Kayıp Fonksiyonu:** `L_total = L_görev (RL) + β * L_VAE`
- `L_VAE` terimi, modelin gözlemi verimli bir şekilde sıkıştırmasını ve yeniden oluşturmasını zorlayarak bir "sadelik baskısı" (Occam's Razor) uygular.

### `CausalAgent`:
- **Mimari:** İki aşamalı:
  1. **SCM Keşif Modülü:** Gözlem ve eylem verilerinden bir Yapısal Nedensel Model (SCM) grafiği öğrenir.
  2. **Do-Planner:** Planlama aşamasında, olası eylem dizilerini SCM üzerinde `do(eylem)` müdahaleleri olarak simüle eder ve en iyi sonucu veren planı seçer.
- **Zorluk:** SCM'yi veriden öğrenmek zordur. İlk deneylerde SCM'nin bilindiği varsayılabilir.

### `WorldModelAgent` (Dreamer-benzeri):
- **Mimari:** Üç ana bileşen:
  1. **Temsil Modeli (RSSM):** Gözlemleri stokastik bir `z` latent durumuna sıkıştırır.
  2. **Geçiş Modeli (RNN):** Latent uzayda bir sonraki durumu (`z_t+1`) tahmin eder (`z_t` ve `eylem_t`'den).
  3. **Aktör-Eleştirmen:** Tamamen latent uzayda ("hayalde") eğitilir.
- **Ölçüm:** "Hayal-gerçek sapması" ve örnek verimliliği (gerçek dünya etkileşimi/performans oranı).

### `FEPAgent` (Active Inference):
- **Mimari:** Bir "üretken model" `p(gözlem, durum)` ve bir "tanıma modeli" `q(durum|gözlem)` içerir.
- **Optimizasyon:** Ajan, hem iç durumlarını (algı) hem de eylemlerini (eylem seçimi) varyasyonel serbest enerjiyi minimize etmek için optimize eder. Eylemler, gelecekteki beklenen serbest enerjiyi (belirsizliği azaltan ve hedeflere ulaştıran) en aza indirecek şekilde seçilir.

---

## 6. Temel Başarı Metrikleri

Ajanların performansı aşağıdaki eksenlerde, SOTA benchmark'larından daha ilkesel metriklerle ölçülecektir:

- **OOD Genelleme:** Eğitim dağıtımından farklı (örn. renk, doku, fizik parametreleri değişmiş) ortamlardaki performans düşüşü.
- **Few-Shot Transfer:** Yeni bir göreve 1-10 örnekle ne kadar hızlı adapte olabildiği.
- **Örnek Verimliliği:** Belirli bir performans seviyesine ulaşmak için gereken ortam etkileşim sayısı.
- **Nedensel Duyarlılık:** Ortamın nedensel yapısına bir müdahale yapıldığında (örn. yerçekimini artırmak), politikanın ne kadar hızlı ve doğru adapte olduğu.
- **Hesaplama/Enerji Maliyeti:** `FLOPs-per-reward` veya benzeri bir metrikle, birim ödüle ulaşmanın hesaplama maliyeti.