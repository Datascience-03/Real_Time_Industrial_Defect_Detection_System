import os
import shutil
import random

random.seed(42)

source_dir = "dataset/raw"
train_dir = "dataset/train"
valid_dir = "dataset/valid"
test_dir = "dataset/test"

train_ratio = 0.8
valid_ratio = 0.1
test_ratio = 0.1

classes = os.listdir(source_dir)

for cls in classes:
    cls_path = os.path.join(source_dir, cls)

    if not os.path.isdir(cls_path):
        continue

    images = os.listdir(cls_path)
    random.shuffle(images)

    total = len(images)

    train_end = int(total * train_ratio)
    valid_end = train_end + int(total * valid_ratio)

    train_images = images[:train_end]
    valid_images = images[train_end:valid_end]
    test_images = images[valid_end:]

    os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(valid_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(test_dir, cls), exist_ok=True)

    for img in train_images:
        shutil.copy(
            os.path.join(cls_path, img),
            os.path.join(train_dir, cls, img)
        )

    for img in valid_images:
        shutil.copy(
            os.path.join(cls_path, img),
            os.path.join(valid_dir, cls, img)
        )

    for img in test_images:
        shutil.copy(
            os.path.join(cls_path, img),
            os.path.join(test_dir, cls, img)
        )

print("Dataset successfully split!")