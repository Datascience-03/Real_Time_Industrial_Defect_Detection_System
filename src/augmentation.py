import os
import cv2
import albumentations as A

# Input and output folder paths
INPUT_DIR = "dataset/raw"
OUTPUT_DIR = "outputs/augmented_images"

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define all augmentation techniques
augmentations = {

    # Rotate image randomly between -30° and +30°
    "rotate": A.Rotate(limit=30, p=1),

    # Flip image horizontally
    "horizontal_flip": A.HorizontalFlip(p=1),

    # Flip image vertically
    "vertical_flip": A.VerticalFlip(p=1),

    # Randomly change brightness and contrast
    "brightness_contrast": A.RandomBrightnessContrast(
        brightness_limit=0.2,
        contrast_limit=0.2,
        p=1
    ),

    # Apply Gaussian Blur
    "gaussian_blur": A.GaussianBlur(
        blur_limit=(3, 5),
        p=1
    ),

    # Add Gaussian Noise
    "gaussian_noise": A.GaussNoise(
        std_range=(0.05, 0.2),
        p=1
    )
}

# Variables to count processed images
original_images = 0
augmented_images = 0

# Loop through each defect class folder
for defect_class in os.listdir(INPUT_DIR):

    input_class_path = os.path.join(INPUT_DIR, defect_class)

    # Skip files and process only folders
    if not os.path.isdir(input_class_path):
        continue

    print(f"\nProcessing Class: {defect_class}")

    # Create corresponding output folder
    output_class_path = os.path.join(OUTPUT_DIR, defect_class)
    os.makedirs(output_class_path, exist_ok=True)

    # Loop through every image in the class folder
    for image_name in os.listdir(input_class_path):

        image_path = os.path.join(input_class_path, image_name)

        # Read image using OpenCV
        image = cv2.imread(image_path)

        # Skip if image cannot be read
        if image is None:
            print(f"Could not read {image_name}")
            continue

        original_images += 1

        # Separate filename and extension
        filename, extension = os.path.splitext(image_name)

        # Apply every augmentation one by one
        for aug_name, transform in augmentations.items():

            # Apply augmentation
            augmented = transform(image=image)

            # Get augmented image
            augmented_image = augmented["image"]

            # Create output filename
            output_filename = f"{aug_name}_{filename}{extension}"

            output_path = os.path.join(
                output_class_path,
                output_filename
            )

            # Save augmented image
            cv2.imwrite(output_path, augmented_image)

            augmented_images += 1

            print(f"Saved -> {output_filename}")

# Display summary after processing all images
print("\nAugmentation Completed Successfully!")
print(f"Original Images Processed : {original_images}")
print(f"Augmented Images Created : {augmented_images}")
print(f"Output Folder : {OUTPUT_DIR}")