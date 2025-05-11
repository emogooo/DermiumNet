import os
import pandas as pd
import shutil

# Dosya yolları
image_folder = r'D:\Github\DermiumNet\datasets\isic2019\images'
csv_file = r'D:\Github\DermiumNet\datasets\isic2019\gt.csv'
output_folder = r'D:\Github\DermiumNet\datasets\isic2019\sorted_images'

# CSV dosyasını oku
df = pd.read_csv(csv_file)

# Her satırda işlem yap
for index, row in df.iterrows():
    image_name = row['image'] + '.jpg'  # Resim adı
    image_path = os.path.join(image_folder, image_name)

    # Sınıf etiketlerini kontrol et
    for label in ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']:
        if row[label] == 1.0:
            # Klasör yolu
            class_folder = os.path.join(output_folder, label)
            
            # Klasör yoksa oluştur
            if not os.path.exists(class_folder):
                os.makedirs(class_folder)
            
            # Resmi kopyala
            shutil.copy(image_path, os.path.join(class_folder, image_name))

print("İşlem tamamlandı!")
