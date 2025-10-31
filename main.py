import torch
import torch.nn as nn

class CNNBlock(nn.Module):
    def __init__(self, giris_kanallari, cikis_kanallari, **kwargs): 
        super(CNNBlock, self).__init__() # CNN bloğunu başlat
        self.konvolusyon = nn.Conv2d(giris_kanallari, cikis_kanallari, bias=False, **kwargs) # Konvolüsyon katmanı
        self.batchnormalizasyon = nn.BatchNorm2d(cikis_kanallari) # Batch normalization katmanı
        self.leakyrelu = nn.LeakyReLU(0.1) # LeakyReLU aktivasyon fonksiyonu

    def forward(self, x):
        return self.leakyrelu(self.batchnormalizasyon(self.konvolusyon(x))) # İleri besleme işlemi

class MiniYolo(nn.Module):
    def __init__(self, S=7, C=1):
        super(MiniYolo, self).__init__()
        self.S = S # Size: grid boyutu
        self.C = C # Classes: sınıf sayısı
        
        self.ozellikler = nn.Sequential(
            CNNBlock(3, 64, kernel_size=7, stride=2, padding=3), nn.MaxPool2d(2, 2),
            CNNBlock(64, 192, kernel_size=3, padding=1), nn.MaxPool2d(2, 2),
            CNNBlock(192, 128, kernel_size=1),
            CNNBlock(128, 256, kernel_size=3, padding=1),
            CNNBlock(256, 256, kernel_size=1),
            CNNBlock(256, 512, kernel_size=3, padding=1), nn.MaxPool2d(2, 2),
            CNNBlock(512, 256, kernel_size=1),
            CNNBlock(256, 512, kernel_size=3, padding=1), nn.MaxPool2d(2, 2),
            CNNBlock(512, 512, kernel_size=1),
            CNNBlock(512, 1024, kernel_size=3, padding=1), nn.MaxPool2d(2, 2)
        ) # Özellik çıkarma katmanları
        
        self.tahminEt = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 496),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(496, S * S * (C + 5)), 
        ) # Tahmin katmanları

    def forward(self, x):
        x = self.ozellikler(x)
        tahminler = self.tahminEt(x)
        return tahminler.reshape(-1, self.S, self.S, self.C + 5) # Çıkışı yeniden şekillendir