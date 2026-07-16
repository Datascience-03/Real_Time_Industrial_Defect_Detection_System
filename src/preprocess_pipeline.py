import cv2
import os
import numpy as np

# PATH CONFIGURATION
INPUT_DIR = 'dataset/raw'
OUTPUT_DIR = 'dataset/augmented'
IMG_SIZE = (640, 640)

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None: return None
    img = cv2.resize(img, IMG_SIZE)
    img_normalized = img.astype(np.float32) / 255.0
    return img_normalized

def run_pipeline():
    print(f"Starting integrated pipeline from {INPUT_DIR}...")
    count = 0
    
    # This 'walks' through all subfolders (crazing, inclusion, etc.)
    for root, dirs, files in os.walk(INPUT_DIR):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                input_path = os.path.join(root, filename)
                
                # Create matching subfolders in augmented
                rel_path = os.path.relpath(root, INPUT_DIR)
                target_dir = os.path.join(OUTPUT_DIR, rel_path)
                if not os.path.exists(target_dir): os.makedirs(target_dir)
                
                output_path = os.path.join(target_dir, filename)

                processed_img = preprocess_image(input_path)
                if processed_img is not None:
                    final_img = (processed_img * 255).astype(np.uint8)
                    cv2.imwrite(output_path, final_img)
                    print(f"Integrated and Processed: {rel_path}/{filename}")
                    count += 1

    print(f"\nPipeline Complete. Processed {count} images into {OUTPUT_DIR}.")

if __name__ == "__main__":
    run_pipeline()