import os
import cv2
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
from tqdm import tqdm

class DataPreprocessor:
    def __init__(self, target_size=(128, 128)):
        self.target_size = target_size
    
    def preprocess_image(self, image_path):
        """Load and preprocess an image"""
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not read image {image_path}")
            return None
        img = cv2.resize(img, self.target_size)
        img = img.astype(np.float32) / 255.0  # Normalize
        return img

    # -------- MRL DATASET PROCESSING --------
    def load_mrl_dataset(self, dataset_path):
        """Load MRL dataset with Open/Closed eyes classes"""
        X, y = [], []
        for label, folder_name in enumerate(["Open_Eyes", "Closed_Eyes"]):
            folder_path = os.path.join(dataset_path, folder_name)
            if os.path.exists(folder_path):
                for img_file in tqdm(os.listdir(folder_path), 
                                     desc=f"Loading {folder_name}"):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(folder_path, img_file)
                        img = self.preprocess_image(img_path)
                        if img is not None:
                            X.append(img)
                            y.append(label)
        X = np.array(X)
        y = tf.keras.utils.to_categorical(np.array(y), 2)
        return shuffle(X, y, random_state=42)

    # -------- YAWN DATASET PROCESSING --------
    def load_yawdd_frames(self, dataset_path):
        """Load YawDD dataset with 4 classes"""
        X, y = [], []
        class_mapping = {
            "Open": 0,
            "Close": 1,
            "Yawn": 2,
            "No_Yawn": 3
        }
        
        for class_name, class_id in class_mapping.items():
            class_path = os.path.join(dataset_path, class_name)
            if os.path.exists(class_path):
                for img_file in tqdm(os.listdir(class_path), 
                                    desc=f"Loading {class_name}"):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_path, img_file)
                        img = self.preprocess_image(img_path)
                        if img is not None:
                            X.append(img)
                            y.append(class_id)
        
        X = np.array(X)
        y = tf.keras.utils.to_categorical(np.array(y), 4)
        return shuffle(X, y, random_state=42)

    # -------- DATA AUGMENTATION --------
    def augment_images(self, X, y, multiplier=2):
        """Apply image augmentation"""
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            brightness_range=(0.8, 1.2),
            horizontal_flip=True,
        )
        
        X_aug, y_aug = [], []
        for i in range(len(X)):
            img = X[i]
            label = y[i]
            X_aug.append(img)
            y_aug.append(label)
            
            # Generate augmented versions
            img_exp = np.expand_dims(img, 0)
            gen = datagen.flow(img_exp, batch_size=1)
            for _ in range(multiplier - 1):
                aug_img = next(gen)[0]
                X_aug.append(aug_img)
                y_aug.append(label)
                
        return np.array(X_aug), np.array(y_aug)
