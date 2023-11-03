import cv2
import os
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import random
import imageio
import imgaug as ia
from imgaug import augmenters as iaa

# Load the original image
original_image = Image.open('WM4.jpg')

# Create a directory to save augmented images
output_dir = 'augmented_dataset5'
os.makedirs(output_dir, exist_ok=True)

# Define the number of augmented samples per operation
num_augmentations = 20

# Initialize a list to store unique augmented images
unique_augmented_images = []

# Define a variety of augmentation operations using Pillow
augmentations_pillow = [
    original_image,
    original_image.rotate(30),
    original_image.rotate(60),
    original_image.transpose(Image.FLIP_LEFT_RIGHT),
    ImageEnhance.Brightness(original_image).enhance(1.5),
    ImageEnhance.Brightness(original_image).enhance(0.7),
    ImageEnhance.Contrast(original_image).enhance(1.2),
    ImageEnhance.Contrast(original_image).enhance(0.8),
    original_image.filter(ImageFilter.GaussianBlur(radius=5)),
    original_image.filter(ImageFilter.EMBOSS),
]

# Initialize imgaug's random seed
ia.seed(4)

# Define a variety of augmentation sequences using imgaug
augmentations_imgaug = [
    iaa.Affine(rotate=(-25, 25)),
    iaa.Affine(rotate=(-45, 45)),
    iaa.AdditiveGaussianNoise(scale=(10, 60)),
    iaa.AdditiveGaussianNoise(scale=(20, 80)),
    iaa.Crop(percent=(0, 0.2)),
    iaa.Crop(percent=(0, 0.4)),
]

# Perform Pillow-based augmentations
for i, augmentation in enumerate(augmentations_pillow):
    for j in range(num_augmentations):
        augmented_image = augmentation.copy()
        output_path = os.path.join(output_dir, f'augmented_pillow_{i}_{j}.jpg')
        augmented_image.save(output_path)
        unique_augmented_images.append(augmented_image)

# Perform imgaug-based augmentations
for k, augmentation_seq in enumerate(augmentations_imgaug):
    for j in range(num_augmentations):
        image = imageio.imread('WM4.jpg')
        image_aug = augmentation_seq(image=image.copy())
        output_path = os.path.join(output_dir, f'augmented_imgaug_{k}_{j}.jpg')
        imageio.imsave(output_path, image_aug)
        unique_augmented_images.append(image_aug)

print(f"Generated {len(unique_augmented_images)} unique augmented images in '{output_dir}'.")