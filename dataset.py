import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class SyntheticDataset(Dataset):
    def __init__(self, resim_adresi, label_dir, S=7, C=1, transform=None):
        self.resim_adresi = resim_adresi # Resimlerin bulunduğu dizin
        self.label_dir = label_dir # Etiket dosyalarının bulunduğu dizin
        self.transform = transform # Görüntü dönüşümleri
        self.S = S # Size: grid boyutu
        self.C = C # Classes: sınıf sayısı
        self.images = os.listdir(resim_adresi) # Resim dosyalarının listesi

    def __len__(self):
        return len(self.images) # Toplam resim sayısı


    def __getitem__(self, index):
        img_name = self.images[index]
        img_path = os.path.join(self.resim_adresi, img_name) 
        
        # Görselin adından etiket dosyasının adını oluştur
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(self.label_dir, label_name)

        # Resmi yükle ve dönüştür
        image = Image.open(img_path).convert("RGB")
        
        # Etiketleri oku
        kutular = []
        with open(label_path) as f:
            for label in f.readlines():
                # .strip() ekleyerek satır sonu karakterlerinden kurtuluyoruz
                sinif_label, x, y, width, height = [
                    float(val) for val in label.strip().split()
                ]
                kutular.append([sinif_label, x, y, width, height])

        # Görüntüye transform uygula
        if self.transform:
            image = self.transform(image)

        # Hedef tensörü oluştur
        label_matrix = torch.zeros((self.S, self.S, self.C + 5)) # 5: x, y, w, h, confidence ve class

        for kutu in kutular:
            sinif_label, x, y, width, height = kutu # unpack et
            sinif_label = int(sinif_label) # sınıf etiketini integer yap
            
            # Hücre indekslerini bul
            i, j = int(self.S * y), int(self.S * x)
            
            # Hücre içi koordinatları bul
            x_cell, y_cell = self.S * x - j, self.S * y - i
            
            # Hücre boşsa bilgforward yaz
            if label_matrix[i, j, 4] == 0:
                # Confidence'i 1 yap
                label_matrix[i, j, 4] = 1
                # Koordinatları yaz
                kutu_koordinatinates = torch.tensor([x_cell, y_cell, width, height])
                label_matrix[i, j, 0:4] = kutu_koordinatinates
                # Sınıf bilgisini one-hot olarak yaz
                label_matrix[i, j, 5 + sinif_label] = 1
                
        return image, label_matrix
