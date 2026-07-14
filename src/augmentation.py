import albumentations as A
import cv2
import os
import glob

def augment_dataset(input_img_dir, input_label_dir, output_img_dir, output_label_dir, num_augs=2):
    # 1. Define the Augmentation Pipeline
    # This handles both the image and the YOLO bounding boxes
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    # Create output directories if they don't exist
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    # 2. Get all image paths
    image_paths = glob.glob(os.path.join(input_img_dir, "*.jpg")) + \
                  glob.glob(os.path.join(input_img_dir, "*.png"))

    print(f"Starting augmentation on {len(image_paths)} images...")

    for img_path in image_paths:
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load corresponding label file (.txt)
        base_name = os.path.basename(img_path).rsplit('.', 1)[0]
        label_path = os.path.join(input_label_dir, f"{base_name}.txt")
        
        bboxes = []
        class_labels = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.split()
                    class_labels.append(int(parts[0]))
                    bboxes.append([float(x) for x in parts[1:]])

        # 3. Apply Augmentations
        for i in range(num_augs):
            try:
                transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
                aug_img = transformed['image']
                aug_bboxes = transformed['bboxes']

                # Save Augmented Image
                save_img_path = os.path.join(output_img_dir, f"{base_name}_aug_{i}.jpg")
                cv2.imwrite(save_img_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))

                # Save Augmented Label
                save_lab_path = os.path.join(output_label_dir, f"{base_name}_aug_{i}.txt")
                with open(save_lab_path, 'w') as f:
                    for cls, box in zip(class_labels, aug_bboxes):
                        f.write(f"{cls} {' '.join(map(str, box))}\n")
            except Exception as e:
                print(f"Skipping an augmentation for {base_name} due to: {e}")

    print("✅ Augmentation complete! Check the 'dataset/augmented' folder.")

if __name__ == "__main__":
    # Define your paths based on your folder structure
    # Make sure these folders exist!
    TRAIN_IMAGES = "dataset/train/images"
    TRAIN_LABELS = "dataset/train/labels"
    
    OUT_IMAGES = "dataset/augmented/images"
    OUT_LABELS = "dataset/augmented/labels"

    # Run the function
    augment_dataset(TRAIN_IMAGES, TRAIN_LABELS, OUT_IMAGES, OUT_LABELS, num_augs=3)