import os
import shutil
import random

# Set seed for reproducibility
random.seed(42)

# Define the dataset path
dataset_dir = "datasets/flowersTestAugment"

# Define the output directories
base_dir = "datasets/flowersTestAugmentSplit"
train_dir = os.path.join(base_dir, "train")
validation_dir = os.path.join(base_dir, "validation")
test_dir = os.path.join(base_dir, "test")

# Define split sizes
train_split = 0.8
validation_split = 0.1
test_split = 0.1

# Create output directories if they don't exist
for directory in [train_dir, validation_dir, test_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Get the class names (subdirectories in Flowers)
classes = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]

for cls in classes:
    cls_dir = os.path.join(dataset_dir, cls)
    images = [img for img in os.listdir(cls_dir) if os.path.isfile(os.path.join(cls_dir, img))]
    random.shuffle(images)  # Shuffle images for random splitting

    total = len(images)
    train_end = int(total * train_split)
    validation_end = train_end + int(total * validation_split)

    train_images = images[:train_end]
    validation_images = images[train_end:validation_end]
    test_images = images[validation_end:]

    # Function to copy images to the respective directory
    def copy_images(image_list, subset_dir):
        cls_subset_dir = os.path.join(subset_dir, cls)
        if not os.path.exists(cls_subset_dir):
            os.makedirs(cls_subset_dir)
        for img in image_list:
            src = os.path.join(cls_dir, img)
            dst = os.path.join(cls_subset_dir, img)
            shutil.copyfile(src, dst)

    # Copy images to train, validation, and test directories
    copy_images(train_images, train_dir)
    copy_images(validation_images, validation_dir)
    copy_images(test_images, test_dir)

print("Dataset successfully split into train, validation, and test sets.")
