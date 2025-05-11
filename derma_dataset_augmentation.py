import os
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# Veri seti klasör yolları
base_folder = r"D:/Github/DermiumNet/datasets/isic2019/sorted_images"
augmented_base_folder = r"D:/Github/DermiumNet/datasets/isic2019/sorted_images_augmented"  # Ana klasör için yeni yol
diseases = ["AK", "BCC", "BKL", "DF", "MEL", "NV", "SCC", "VASC"]
target_count = 2000  # En çok bulunan sınıfın (nv) görüntü sayısı

# Ana klasörü oluştur
os.makedirs(augmented_base_folder, exist_ok=True)

# ImageDataGenerator oluşturma
datagen = ImageDataGenerator(rescale=1.0 / 255, rotation_range=40, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.1, zoom_range=0.1, horizontal_flip=True, vertical_flip=True, fill_mode="nearest")

# Her hastalık türü için resimlerin kopyalanması ve artırılması
for disease in diseases:
    counter = 0
    print(f"Processing {disease}...")
    disease_folder = os.path.join(base_folder, disease)
    images = [os.path.join(disease_folder, f) for f in os.listdir(disease_folder) if f.endswith(".jpg")]
    image_count = len(images)

    # Oluşturulacak klasörü kontrol et
    augmented_folder = os.path.join(augmented_base_folder, f"{disease}")
    os.makedirs(augmented_folder, exist_ok=True)

    # Orijinal görüntüleri yeni klasöre kopyala
    for img_path in images:
        if counter < target_count:
            shutil.copy(img_path, augmented_folder)
            counter += 1
        else:
            break
    if counter == target_count:
        counter = 0
        continue

    how_many_left = target_count - counter
    for i in range(how_many_left):
        img_index = i % image_count  # Mevcut görüntülerin döngüsü
        img_path = images[img_index]
        img_name = os.path.basename(img_path)

        # Görüntüyü yükle ve artır
        img = load_img(img_path)
        img = img_to_array(img)
        img = img.reshape((1,) + img.shape)

        # Artırılmış görüntüyü kaydet
        save_prefix = f"aug_{i}_{img_name}"
        for batch in datagen.flow(img, batch_size=1, save_to_dir=augmented_folder, save_prefix=save_prefix, save_format="jpg"):
            break  # Her görüntü için bir kez döngü yap

print("Data augmentation complete.")
