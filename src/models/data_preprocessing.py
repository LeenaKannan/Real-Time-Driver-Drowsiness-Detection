# src/models/data_preprocessing.py
import os
import cv2
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt

class DataPreprocessor:
    def __init__(self, target_size=(128, 128)):
        self.target_size = target_size
        
    def preprocess_image(self, image_path):
        """Enhanced image preprocessing with error handling"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None
                
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize
            img = cv2.resize(img, self.target_size)
            
            # Normalize to [0, 1]
            img = img.astype(np.float32) / 255.0
            
            return img
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def load_mrl_dataset(self, dataset_path):
        """Load MRL dataset with enhanced error handling"""
        X, y = [], []
        
        # MRL dataset structure: Open_Eyes, Closed_Eyes
        class_folders = {
            "Open_Eyes": 0,    # Awake
            "Closed_Eyes": 1   # Drowsy
        }
        
        for folder_name, label in class_folders.items():
            folder_path = os.path.join(dataset_path, folder_name)
            
            if not os.path.exists(folder_path):
                print(f"Warning: {folder_path} not found")
                continue
                
            image_files = [f for f in os.listdir(folder_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            print(f"Loading {len(image_files)} images from {folder_name}")
            
            for img_file in tqdm(image_files, desc=f"Processing {folder_name}"):
                img_path = os.path.join(folder_path, img_file)
                img = self.preprocess_image(img_path)
                
                if img is not None:
                    X.append(img)
                    y.append(label)
        
        if len(X) == 0:
            raise ValueError("No valid images found in MRL dataset")
            
        X = np.array(X)
        y = tf.keras.utils.to_categorical(np.array(y), 2)
        
        print(f"MRL Dataset loaded: {X.shape[0]} samples")
        print(f"Class distribution: {np.bincount(np.argmax(y, axis=1))}")
        
        return shuffle(X, y, random_state=42)
    
    def load_yawdd_dataset(self, dataset_path):
        """Load YawDD dataset and convert to binary classification"""
        X, y = [], []
        
        # YawDD dataset structure mapping to binary
        class_mapping = {
            "Open": 0,      # Awake
            "Close": 1,     # Drowsy
            "Yawn": 1,      # Drowsy (yawning indicates drowsiness)
            "No_Yawn": 0    # Awake
        }
        
        for class_name, binary_label in class_mapping.items():
            class_path = os.path.join(dataset_path, class_name)
            
            if not os.path.exists(class_path):
                print(f"Warning: {class_path} not found")
                continue
                
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            print(f"Loading {len(image_files)} images from {class_name}")
            
            for img_file in tqdm(image_files, desc=f"Processing {class_name}"):
                img_path = os.path.join(class_path, img_file)
                img = self.preprocess_image(img_path)
                
                if img is not None:
                    X.append(img)
                    y.append(binary_label)
        
        if len(X) == 0:
            raise ValueError("No valid images found in YawDD dataset")
            
        X = np.array(X)
        y = tf.keras.utils.to_categorical(np.array(y), 2)
        
        print(f"YawDD Dataset loaded: {X.shape[0]} samples")
        print(f"Class distribution: {np.bincount(np.argmax(y, axis=1))}")
        
        return shuffle(X, y, random_state=42)
    
    def combine_datasets(self, datasets):
        """Combine multiple datasets"""
        X_combined = []
        y_combined = []
        
        for X, y in datasets:
            X_combined.append(X)
            y_combined.append(y)
        
        X_final = np.vstack(X_combined)
        y_final = np.vstack(y_combined)
        
        return shuffle(X_final, y_final, random_state=42)
    
    def visualize_dataset(self, X, y, num_samples=16):
        """Visualize dataset samples"""
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        axes = axes.ravel()
        
        class_names = ['Awake', 'Drowsy']
        
        for i in range(min(num_samples, len(X))):
            axes[i].imshow(X[i])
            axes[i].set_title(f'{class_names[np.argmax(y[i])]}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
