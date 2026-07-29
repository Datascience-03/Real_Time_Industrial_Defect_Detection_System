from PIL import Image
import os

dataset_path = "dataset/raw"

corrupted = []

for class_name in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_name)

    if not os.path.isdir(class_path):
        continue

    for image_name in os.listdir(class_path):
        image_path = os.path.join(class_path, image_name)

        try:
            with Image.open(image_path) as img:
                img.verify()   # Verify image integrity
        except Exception:
            corrupted.append(image_path)

if len(corrupted) == 0:
    print("✅ No corrupted images found.")
else:
    print(f"❌ Found {len(corrupted)} corrupted image(s):")
    for img in corrupted:
        print(img)