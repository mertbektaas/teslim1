import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import SyntheticDataset
from main import MiniYolo
from loss import YoloLoss


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# augmentasyonlar
transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1), # Renk değişiklikleri
    transforms.RandomHorizontalFlip(p=0.5), # Görüntüyü %50 ihtimalle yatayda çevirir
    transforms.RandomRotation(degrees=15), # Görüntüyü +/- 15 derece döndürür

    
    transforms.ToTensor()
])

# dataset oluştur
train_dataset = SyntheticDataset(
    resim_adresi="dataset/images", 
    label_dir="dataset/labels", 
    S=7, C=1, 
    transform=transform
)
# dataseti yükle
train_loader = DataLoader(
    dataset=train_dataset, 
    batch_size=8, 
    shuffle=True
)

model = MiniYolo(S=7, C=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
kayip_fn = YoloLoss(S=7, C=1)

print("Eğitim başlıyor...")


epoch_num = 5

for epoch in range(epoch_num):
    total_kayip = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        tahminler = model(images)
        kayip = kayip_fn(tahminler, labels)
        
        optimizer.zero_grad()
        kayip.backward()
        optimizer.step()
        
        total_kayip += kayip.item()
        
        if batch_idx % 3 == 0:
            print(f"Epoch [{epoch+1}/{epoch_num}], Batch {batch_idx+1}/{len(train_loader)}, kayip: {kayip.item():.4f}")
    
    avg_kayip = total_kayip / len(train_loader)
    print(f">>> Epoch {epoch+1} Ortalama kayip: {avg_kayip:.4f}")
    
    if avg_kayip < 0.3:
        torch.save(model.state_dict(), "miniyolo_best.pth")
        print(f"En iyi model kaydedildi! kayip: {avg_kayip:.4f}")

print("\nEğitim tamamlandı!")
torch.save(model.state_dict(), "miniyolo.pth")
print("Model 'miniyolo.pth' olarak kaydedildi.")


print("\n--- Hızlı Test ---")
model.eval()
with torch.no_grad():
    # Eğitim setinden ilk resmi al 
    test_dataset_no_aug = SyntheticDataset(
        resim_adresi="dataset/images", 
        label_dir="dataset/labels", 
        S=7, C=1, 
        transform=transforms.Compose([transforms.Resize((448, 448)), transforms.ToTensor()])
    )
    test_img, test_label = test_dataset_no_aug[0]
    test_img = test_img.unsqueeze(0).to(device)
    pred = model(test_img).cpu()
    pred[..., 0:2] = torch.sigmoid(pred[..., 0:2])
    pred[..., 2:4] = torch.sigmoid(pred[..., 2:4])
    pred[..., 4:] = torch.sigmoid(pred[..., 4:])
    
    confidence_map = pred[0, :, :, 4]
    max_conf = confidence_map.max().item()
    print(f"Maksimum confidence: {max_conf:.4f}")
    if max_conf > 0.5:
        print("Yeterli")
    else:
        print("Yetersiz")
