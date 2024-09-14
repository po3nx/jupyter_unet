import os
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img

def load_images(image_dir, mask_dir, target_size=(128, 128)):
    images = []
    masks = []
    filenames = os.listdir(image_dir)
    for filename in filenames:
        img_path = os.path.join(image_dir, filename)
        mask_path = os.path.join(mask_dir, filename)
        img = load_img(img_path, target_size=target_size)
        mask = load_img(mask_path, target_size=target_size, color_mode="grayscale")
        img = img_to_array(img)
        mask = img_to_array(mask)
        images.append(img)
        masks.append(mask)
    images = np.array(images, dtype="float32")
    masks = np.array(masks, dtype="float32")
    masks /= 255.0  # Normalize masks to [0, 1]
    return images, masks, filenames

def save_preprocessed_data(images, masks, processed_dir):
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    np.save(os.path.join(processed_dir, 'images.npy'), images)
    np.save(os.path.join(processed_dir, 'masks.npy'), masks)

def load_preprocessed_data(processed_dir):
    images = np.load(os.path.join(processed_dir, 'images.npy'))
    masks = np.load(os.path.join(processed_dir, 'masks.npy'))
    return images, masks