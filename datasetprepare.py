import os
from PIL import Image, ImageDraw
import random

def create_synthetic_dataset(output_dir="dataset", num_images=100, img_size=(448, 448)):
    print(f"'{output_dir}' içinde {num_images} resimlik Gelişmiş Veri Seti oluşturuluyor...")
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    sinif_id = 0
    
    
    obje_colors = ["red", "green", "blue", "purple", "black", "orange", "yellow"]

    for i in range(num_images):
      
        bg_r = random.randint(220, 255)
        bg_g = random.randint(220, 255)
        bg_b = random.randint(220, 255)
        img = Image.new("RGB", img_size, (bg_r, bg_g, bg_b))
        draw = ImageDraw.Draw(img)
      
        obj_width = random.randint(30, 150)
        obj_height = random.randint(30, 150) 
        
        x1 = random.randint(0, img_size[0] - obj_width)
        y1 = random.randint(0, img_size[1] - obj_height)
        x2 = x1 + obj_width
        y2 = y1 + obj_height
 
        chosen_color = random.choice(obje_colors)
        
  
        draw.rectangle((x1, y1, x2, y2), fill=chosen_color)
        
        img_path = os.path.join(images_dir, f"img_{i}.png")
        img.save(img_path)


        x_center = ((x1 + x2) / 2) / img_size[0]
        y_center = ((y1 + y2) / 2) / img_size[1]
        norm_width = obj_width / img_size[0]
        norm_height = obj_height / img_size[1]
        label_path = os.path.join(labels_dir, f"img_{i}.txt")
        with open(label_path, "w") as f:
            f.write(f"{sinif_id} {x_center} {y_center} {norm_width} {norm_height}\n")
            
    print("Veri seti oluşturma tamamlandı.")

if __name__ == '__main__':

    create_synthetic_dataset(output_dir="dataset", num_images=1000)
