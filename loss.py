import torch
import torch.nn as nn

class YoloLoss(nn.Module):
    def __init__(self, S=7, C=1, lambda_koordinat=5, lambda_objeyok=0.5):
        super(YoloLoss, self).__init__()
        self.okh = nn.MSELoss(reduction="sum") # Ortalama kare hata kaybı
        self.S = S # Size: grid boyutu
        self.C = C # Classes: sınıf sayısı
        self.lambda_koordinat = lambda_koordinat # Koordinat kaybı için ağırlık
        self.lambda_objeyok = lambda_objeyok # Nesne olmayan hücreler için ağırlık

    def forward(self, tahminler, hedef):
        # Kayıp hesaplamadan önce tahminleri doğru formata getiriyoruz
        # Sigmoid'le değerleri normalize et (0-1 arası)
        tahminler_clone = tahminler.clone() # Tahminleri clonela
        tahminler_clone[..., 0:2] = torch.sigmoid(tahminler_clone[..., 0:2])  # x, y
        tahminler_clone[..., 2:4] = torch.sigmoid(tahminler_clone[..., 2:4])  # w, h
        tahminler_clone[..., 4:] = torch.sigmoid(tahminler_clone[..., 4:])    # güven, sınıf
 
        kutu_mevcut = hedef[..., 4] > 0 # Nesne var mı maskesi
        
      
        kutu_tahminler = kutu_mevcut.unsqueeze(-1) * tahminler_clone # Nesne olan kutu tahminleri
        kutu_hedefler = kutu_mevcut.unsqueeze(-1) * hedef # Nesne olan kutu hedefleri
        
        
        koordinat_kayip = self.okh(kutu_tahminler[..., :2], kutu_hedefler[..., :2]) # x,y için kayıp
        
        
        koordinat_kayip += self.okh(kutu_tahminler[..., 2:4], kutu_hedefler[..., 2:4]) # w,h için kayıp

        
        obje_kayip = self.okh(kutu_tahminler[..., 4], kutu_hedefler[..., 4]) # Nesne olan hücreler için kayıp
        no_obje_kayip = self.okh(
            (1 - kutu_mevcut.float()) * tahminler_clone[..., 4], 
            (1 - kutu_mevcut.float()) * hedef[..., 4]
        ) # Nesne olmayan hücreler için kayıp


        sinif_kayip = self.okh(kutu_tahminler[..., 5:], kutu_hedefler[..., 5:]) # Sınıf kaybı

        kayip = (
            self.lambda_koordinat * koordinat_kayip + 
            obje_kayip + 
            self.lambda_objeyok * no_obje_kayip + 
            sinif_kayip
        ) # Toplam kayıp

        return kayip / tahminler.shape[0]