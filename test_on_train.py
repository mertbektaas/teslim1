import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from collections import Counter
import numpy as np

from main import MiniYolo
from dataset import SyntheticDataset

def intersection_over_union(kutues_preds, kutues_labels):
    kutu1_x1 = kutues_preds[..., 0:1] - kutues_preds[..., 2:3] / 2
    kutu1_y1 = kutues_preds[..., 1:2] - kutues_preds[..., 3:4] / 2
    kutu1_x2 = kutues_preds[..., 0:1] + kutues_preds[..., 2:3] / 2
    kutu1_y2 = kutues_preds[..., 1:2] + kutues_preds[..., 3:4] / 2
    
    kutu2_x1 = kutues_labels[..., 0:1] - kutues_labels[..., 2:3] / 2
    kutu2_y1 = kutues_labels[..., 1:2] - kutues_labels[..., 3:4] / 2
    kutu2_x2 = kutues_labels[..., 0:1] + kutues_labels[..., 2:3] / 2
    kutu2_y2 = kutues_labels[..., 1:2] + kutues_labels[..., 3:4] / 2
    
    x1 = torch.max(kutu1_x1, kutu2_x1)
    y1 = torch.max(kutu1_y1, kutu2_y1)
    x2 = torch.min(kutu1_x2, kutu2_x2)
    y2 = torch.min(kutu1_y2, kutu2_y2)
    
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    
    kutu1_area = abs((kutu1_x2 - kutu1_x1) * (kutu1_y2 - kutu1_y1))
    kutu2_area = abs((kutu2_x2 - kutu2_x1) * (kutu2_y2 - kutu2_y1))
    union = kutu1_area + kutu2_area - intersection + 1e-6
    
    return intersection / union

def mean_average_precision(pred_kutues, true_kutues, iou_threshold=0.5, num_classes=1):
    average_precisions = []
    epsilon = 1e-6
    true_positives_iou_list = []

    for c in range(num_classes):
        detections = [kutu for kutu in pred_kutues if kutu[1] == c]
        ground_truths = [kutu for kutu in true_kutues if kutu[1] == c]
        
        amount_bkutues = Counter(gt[0] for gt in ground_truths)
        for key, val in amount_bkutues.items():
            amount_bkutues[key] = torch.zeros(val)
        
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros(len(detections))
        FP = torch.zeros(len(detections))
        total_true_bkutues = len(ground_truths)
        
        if total_true_bkutues == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            ground_truth_img = [bkutu for bkutu in ground_truths if bkutu[0] == detection[0]]
            best_iou = 0
            best_gt_idx = -1
            
            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]), 
                    torch.tensor(gt[3:])
                ).item()
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
            
            if best_iou > iou_threshold:
                if amount_bkutues[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_bkutues[detection[0]][best_gt_idx] = 1
                    true_positives_iou_list.append(best_iou)
                else:
                    FP[detection_idx] = 1
            else:
                FP[detection_idx] = 1
            
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bkutues + epsilon)
        precisions = torch.div(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        average_precisions.append(torch.trapz(precisions, recalls))

    final_ap = sum(ap for ap in average_precisions) / (len(average_precisions) + epsilon)
    return final_ap, true_positives_iou_list

if __name__ == '__main__':
        # Colab (veya herhangi bir NVIDIA GPU'lu sistem) için doğru cihaz belirleme
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        print(f"CUDA (NVIDIA GPU) kullanılıyor. Cihaz: {torch.cuda.get_device_name(0)}")
    else:
        print("CPU kullanılıyor.")
    
    MODEL_PATH = "miniyolo.pth"
    S = 7
    C = 1
    IOU_THRESHOLD = 0.5
    CONFIDENCE_THRESHOLD = 0.4

    # Model yükle
    model = MiniYolo(S=S, C=C).to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # EĞİTİM VERİSİNİ KULLAN
    transform = transforms.Compose([
        transforms.Resize((448, 448)), 
        transforms.ToTensor()
    ])
    train_dataset = SyntheticDataset(
        resim_adresi="dataset/images",  # TEST DEĞİL, TRAIN!
        label_dir="dataset/labels", 
        S=S, 
        C=C, 
        transform=transform
    )
    train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=False)
    
    all_pred_kutues = []
    all_true_kutues = []
    img_idx = 0

    with torch.no_grad():
        for images, labels in train_loader:
            images = images.to(device)
            tahminler = model(images).cpu()
            
            # Gerçek kutuları topla
            for i in range(S):
                for j in range(S):
                    if labels[0, i, j, 4] == 1:
                        x = labels[0, i, j, 0].item()
                        y = labels[0, i, j, 1].item()
                        w = labels[0, i, j, 2].item()
                        h = labels[0, i, j, 3].item()
                        all_true_kutues.append([
                            img_idx, 0, 1.0, 
                            (j + x) / S, (i + y) / S, w, h
                        ])
            
            # Tahminleri normalize et
            tahminler[..., 0:2] = torch.sigmoid(tahminler[..., 0:2])
            tahminler[..., 2:4] = torch.sigmoid(tahminler[..., 2:4])
            tahminler[..., 4:] = torch.sigmoid(tahminler[..., 4:])
            
            # Tahmin edilen kutuları topla
            for i in range(S):
                for j in range(S):
                    confidence = tahminler[0, i, j, 4].item()
                    if confidence > CONFIDENCE_THRESHOLD:
                        x = tahminler[0, i, j, 0].item()
                        y = tahminler[0, i, j, 1].item()
                        w = tahminler[0, i, j, 2].item()
                        h = tahminler[0, i, j, 3].item()
                        all_pred_kutues.append([
                            img_idx, 0, confidence, 
                            (j + x) / S, (i + y) / S, abs(w), abs(h)
                        ])
            
            img_idx += 1
            
            # İlk 20 resim yeterli
            if img_idx >= 20:
                break

    # mAP hesapla
    ap, tp_ious = mean_average_precision(
        all_pred_kutues, 
        all_true_kutues, 
        iou_threshold=IOU_THRESHOLD
    )
    avg_iou_of_tps = np.mean(tp_ious) if len(tp_ious) > 0 else 0.0

    print("\n" + "="*50)
    print("--- EĞİTİM VERİSİNDE Değerlendirme ---")
    print("="*50)
    print(f"Toplam Görüntü                     : {img_idx}")
    print(f"Toplam Gerçek Kutu                 : {len(all_true_kutues)}")
    print(f"Toplam Tahmin Edilen Kutu          : {len(all_pred_kutues)}")
    print(f"Average Precision (AP) @{IOU_THRESHOLD} IoU  : {ap.item():.6f}")
    print(f"Doğru Tahminlerin Ortalama IoU     : {avg_iou_of_tps:.6f}")
    print("="*50)