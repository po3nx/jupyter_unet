import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose
from tensorflow.keras.models import Model, load_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
from skimage.measure import label, regionprops
from tensorflow.keras.preprocessing.image import img_to_array, load_img, save_img
from utils.data_utils import load_images, save_preprocessed_data, load_preprocessed_data

class ImageSegmentationModel:
    def __init__(self, image_dir, mask_dir, model_path='model.h5', processed_dir='data/processed', results_dir='results/segmentation_results'):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.model_path = model_path
        self.processed_dir = processed_dir
        self.results_dir = results_dir
        self.model = self.load_or_create_model()
    
    def load_or_create_model(self):
        if os.path.exists(self.model_path):
            print(f"Loading existing model from {self.model_path}")
            try:
                model = load_model(self.model_path, compile=False)
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                return model
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Creating a new model instead.")
                return self.create_model()
        else:
            print("No existing model found, creating a new one.")
            return self.create_model()

    def create_model(self):
        inputs = Input((128, 128, 3))
        c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
        p1 = MaxPooling2D((2, 2))(c1)
        c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
        p2 = MaxPooling2D((2, 2))(c2)
        c3 = Conv2D(64, (3, 3), activation='relu', padding='same')(p2)
        p3 = MaxPooling2D((2, 2))(c3)
        c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(p3)
        p4 = MaxPooling2D((2, 2))(c4)
        c5 = Conv2D(256, (3, 3), activation='relu', padding='same')(p4)
        u6 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(c5)
        u6 = concatenate([u6, c4])
        c6 = Conv2D(128, (3, 3), activation='relu', padding='same')(u6)
        u7 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(c6)
        u7 = concatenate([u7, c3])
        c7 = Conv2D(64, (3, 3), activation='relu', padding='same')(u7)
        u8 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(c7)
        u8 = concatenate([u8, c2])
        c8 = Conv2D(32, (3, 3), activation='relu', padding='same')(u8)
        u9 = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same')(c8)
        u9 = concatenate([u9, c1], axis=3)
        c9 = Conv2D(16, (3, 3), activation='relu', padding='same')(u9)
        outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
        model = Model(inputs=[inputs], outputs=[outputs])
        return model

    def train(self, test_size=0.1, batch_size=32, epochs=30):
        if os.path.exists(os.path.join(self.processed_dir, 'images.npy')) and os.path.exists(os.path.join(self.processed_dir, 'masks.npy')):
            images, masks = load_preprocessed_data(self.processed_dir)
        else:
            images, masks, _ = load_images(self.image_dir, self.mask_dir)
            save_preprocessed_data(images, masks, self.processed_dir)
        
        X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=test_size, random_state=42)
        print("Training model...")
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        history = self.model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=batch_size, epochs=epochs)
        self.model.save(self.model_path)
        print(f"Model saved to {self.model_path}")
        return history

    def plot_history(self, history):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.legend()
        plt.title('Loss Over Epochs')
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.legend()
        plt.title('Accuracy Over Epochs')
        plt.show()

    def test_model(self, image_path):
        img = load_img(image_path, target_size=(128, 128))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        pred = self.model.predict(img)
        pred = (pred > 0.5).astype(np.uint8)
        pred = np.squeeze(pred)
        
        plt.imshow(np.squeeze(img.astype('uint8')))
        plt.title('Original Image')
        plt.show()
        
        plt.imshow(pred, cmap='gray')
        plt.title('Predicted Mask')
        plt.show()
        
        # Save the predicted mask
        result_path = os.path.join(self.results_dir, f"predicted_mask_{os.path.basename(image_path)}")
        pred_3d = np.expand_dims(pred, axis=-1) 
        pred_3d = np.repeat(pred_3d, 3, axis=-1)
        save_img(result_path, pred_3d)
        print(f"Predicted mask saved to {result_path}")

    def test_random_images(self, num_images=9):
        images, masks, filenames = load_images(self.image_dir, self.mask_dir)
        selected_indices = random.sample(range(len(images)), num_images)
        fig, axs = plt.subplots(num_images, 4, figsize=(20, num_images * 4))
        for i, idx in enumerate(selected_indices):
            img = np.expand_dims(images[idx], axis=0)
            pred = self.model.predict(img)
            pred_threshold = (pred > 0.5).astype(np.uint8)
            pred_threshold = np.squeeze(pred_threshold)
            labeled_mask = label(pred_threshold)
            regions = regionprops(labeled_mask)
            centroids = [region.centroid for region in regions]
            axs[i, 0].imshow(np.squeeze(images[idx].astype('uint8')))
            axs[i, 0].title.set_text('Original Image')
            axs[i, 1].imshow(np.squeeze(masks[idx]), cmap='gray')
            axs[i, 1].title.set_text('Original Mask')
            axs[i, 2].imshow(pred_threshold, cmap='gray')
            axs[i, 2].title.set_text('Predicted Mask')
            axs[i, 3].imshow(np.squeeze(images[idx].astype('uint8')), cmap='gray')
            for centroid in centroids:
                axs[i, 3].scatter(centroid[1], centroid[0], c='red', s=10)
            axs[i, 3].title.set_text(f'Centroids (Count: {len(centroids)})')
            for ax in axs[i]:
                ax.axis('off')
            
            # Save the predicted mask
            result_path = os.path.join(self.results_dir, f"predicted_mask_{filenames[idx]}")
            pred_threshold_3d = np.expand_dims(pred_threshold, axis=-1)
            pred_threshold_3d = np.repeat(pred_threshold_3d, 3, axis=-1) 
            save_img(result_path, pred_threshold_3d)
            print(f"Predicted mask saved to {result_path}")

        plt.show()
    
    def visualize_and_count_centers(self, image_path):
        img = load_img(image_path, target_size=(128, 128))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        pred = self.model.predict(img)
        pred_threshold = (pred > 0.5).astype(np.uint8)
        pred_threshold = np.squeeze(pred_threshold)
        
        plt.imshow(pred_threshold, cmap='gray')
        plt.title('Predicted Mask')
        plt.show()
        
        labeled_mask = label(pred_threshold)
        regions = regionprops(labeled_mask)
        centroids = [region.centroid for region in regions]
        
        plt.figure(figsize=(6, 6))
        plt.imshow(np.squeeze(img.astype('uint8')), cmap='gray')
        
        for centroid in centroids:
            plt.scatter(centroid[1], centroid[0], c='red', s=10)
        
        plt.title(f'Centers of Predicted Regions (Total: {len(centroids)})')
        plt.axis('off')
        plt.show()

        # Save the predicted mask with centroids
        result_path = os.path.join(self.results_dir, f"predicted_mask_with_centroids_{os.path.basename(image_path)}")
        pred_threshold_3d = np.expand_dims(pred_threshold, axis=-1)
        pred_threshold_3d = np.repeat(pred_threshold_3d, 3, axis=-1) 
        save_img(result_path, pred_threshold_3d)
        print(f"Predicted mask with centroids saved to {result_path}")

        return centroids, len(centroids)