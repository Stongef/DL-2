# Evcil Hayvan Irkı Sınıflandırma (Oxford-IIIT Pet + ResNet-50)

Computer Vision ödevi: ImageNet ön eğitimli ResNet-50 ile 37 kedi/köpek ırkını sınıflandıran derin öğrenme uygulaması.

- **Görev:** Görüntü sınıflandırma (37 sınıf)
- **Veri seti:** [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/)
- **Model:** ResNet-50 (transfer learning, son katman 37 sınıfa uyarlandı)
- **Uygulama:** Gradio

## Sonuçlar (test kümesi, 3669 görüntü)

| Metrik | Değer |
|---|---|
| Accuracy | 0.9193 |
| Macro Precision / Recall / F1 | 0.9212 / 0.9187 / 0.9179 |
| Eğitim süresi | 5.5 dk (Tesla T4, Google Colab) |
| Tahmin süresi | 8.2 ms / görüntü (GPU) |

Eğitim ayarları: lr=1e-4, batch=32, 8 epoch, AdamW (weight decay 1e-4), cosine LR schedule, CrossEntropyLoss, giriş 224x224x3.

## Dosyalar

| Dosya | Açıklama |
|---|---|
| `CV_Odev_Pet_ResNet50.ipynb` | Veri hazırlama, eğitim, değerlendirme ve test (Colab) |
| `app.py` | Gradio uygulaması |
| `pet_resnet50.pth` | Eğitilmiş model ağırlıkları |
| `classes.json` | Sınıf isimleri |
| `requirements.txt` | Gerekli paketler |
| `egitim_egrileri.png`, `confusion_matrix.png`, `gercek_test.png` | Eğitim eğrileri, confusion matrix, telefon fotoğrafı testi |

## Uygulamayı çalıştırma

```bash
pip install -r requirements.txt
python app.py
```

Ardından tarayıcıda `http://127.0.0.1:7860` adresini aç.
