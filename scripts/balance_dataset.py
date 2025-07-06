# scripts/balance_dataset.py
import cv2
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator

class DatasetBalancer:
    def __init__(self, dataset_path, target_samples_per_class=2000):
        self.dataset_path = dataset_path
        self.target_samples = target_samples_per_class
        
    def balance_classes(self):
        """Balance awake and drowsy classes to equal samples"""
        
        # Count existing samples
        awake_path = os.path.join(self.dataset_path, 'awake')
        drowsy_path = os.path.join(self.dataset_path, 'drowsy')
        
        awake_files = [f for f in os.listdir(awake_path) if f.endswith('.jpg')]
        drowsy_files = [f for f in os.listdir(drowsy_path) if f.endswith('.jpg')]
        
        print(f"Current samples - Awake: {len(awake_files)}, Drowsy: {len(drowsy_files)}")
        
        # Augment the class with fewer samples
        if len(awake_files) < self.target_samples:
            self.augment_class(awake_path, self.target_samples - len(awake_files))
            
        if len(drowsy_files) < self.target_samples:
            self.augment_class(drowsy_path, self.target_samples - len(drowsy_files))
    
    def augment_class(self, class_path, needed_samples):
        """Augment a specific class to reach target samples"""
        
        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            brightness_range=[0.8, 1.2],
            horizontal_flip=True,
            zoom_range=0.1,
            fill_mode='nearest'
        )
        
        existing_files = [f for f in os.listdir(class_path) if f.endswith('.jpg')]
        
        for i in range(needed_samples):
            # Select random existing image
            source_file = np.random.choice(existing_files)
            source_path = os.path.join(class_path, source_file)
            
            # Load and augment
            img = cv2.imread(source_path)
            img = cv2.resize(img, (128, 128))
            img = np.expand_dims(img, axis=0)
            
            # Generate augmented image
            aug_iter = datagen.flow(img, batch_size=1)
            aug_img = next(aug_iter)[0].astype(np.uint8)
            
            # Save augmented image
            aug_filename = f"aug_{i:04d}_{source_file}"
            cv2.imwrite(os.path.join(class_path, aug_filename), aug_img)
            
        print(f"Generated {needed_samples} augmented samples for {class_path}")

# Usage
balancer = DatasetBalancer("data/balanced_drowsiness", target_samples_per_class=2000)
balancer.balance_classes()
