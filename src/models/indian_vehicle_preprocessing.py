# src/models/indian_vehicle_preprocessing.py
import os
import cv2
import numpy as np
from tqdm import tqdm
import zipfile
from sklearn.preprocessing import LabelEncoder

class IndianVehiclePreprocessor:
    def __init__(self, dataset_path, output_path):
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.target_size = (128, 128)
        self.label_encoder = LabelEncoder()
        
    def extract_dataset(self):
        """Extract Indian vehicle dataset if it's zipped"""
        if self.dataset_path.endswith('.zip'):
            print("Extracting Indian vehicle dataset...")
            with zipfile.ZipFile(self.dataset_path, 'r') as zip_ref:
                zip_ref.extractall(os.path.dirname(self.dataset_path))
            self.dataset_path = self.dataset_path.replace('.zip', '')
    
    def process_vehicle_dataset(self):
        """Process Indian vehicle dataset for vehicle type classification"""
        self.extract_dataset()
        
        X_data = []
        y_data = []
        class_names = []
        
        # Assume dataset structure: dataset/class_name/images
        for class_name in os.listdir(self.dataset_path):
            class_path = os.path.join(self.dataset_path, class_name)
            
            if not os.path.isdir(class_path):
                continue
                
            class_names.append(class_name)
            print(f"Processing class: {class_name}")
            
            for img_file in tqdm(os.listdir(class_path)):
                if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                    
                img_path = os.path.join(class_path, img_file)
                image = cv2.imread(img_path)
                
                if image is None:
                    continue
                    
                # Preprocess image
                image = cv2.resize(image, self.target_size)
                image = image.astype(np.float32) / 255.0
                
                X_data.append(image)
                y_data.append(class_name)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y_data)
        y_onehot = np.eye(len(class_names))[y_encoded]
        
        # Convert to numpy arrays
        X_data = np.array(X_data)
        
        # Save processed data
        os.makedirs(self.output_path, exist_ok=True)
        np.savez_compressed(
            os.path.join(self.output_path, 'indian_vehicle_processed.npz'),
            X=X_data, y=y_onehot, classes=class_names
        )
        
        print(f"Processed vehicle data: {X_data.shape[0]} samples, {len(class_names)} classes")
        return X_data, y_onehot, class_names

if __name__ == "__main__":
    preprocessor = IndianVehiclePreprocessor(
        dataset_path="data/raw/indian_vehicle_dataset.zip",
        output_path="data/processed"
    )
    preprocessor.process_vehicle_dataset()
