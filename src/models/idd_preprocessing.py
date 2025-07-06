# src/models/idd_preprocessing.py
import os
import json
import cv2
import numpy as np
from tqdm import tqdm
import zipfile
from pathlib import Path

class IDDPreprocessor:
    def __init__(self, dataset_path, output_path):
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.target_size = (128, 128)
        
    def extract_dataset(self):
        """Extract IDD dataset if it's zipped"""
        if self.dataset_path.endswith('.zip'):
            print("Extracting IDD dataset...")
            with zipfile.ZipFile(self.dataset_path, 'r') as zip_ref:
                zip_ref.extractall(os.path.dirname(self.dataset_path))
            self.dataset_path = self.dataset_path.replace('.zip', '')
    
    def process_idd_for_drowsiness(self):
        """Process IDD dataset for drowsiness detection (focus on driver/person detection)"""
        self.extract_dataset()
        
        X_data = []
        y_data = []
        
        splits = ['train', 'val', 'test']
        
        for split in splits:
            img_dir = os.path.join(self.dataset_path, 'leftImg8bit', split)
            gt_dir = os.path.join(self.dataset_path, 'gtFine', split)
            
            if not os.path.exists(img_dir):
                continue
                
            print(f"Processing {split} split...")
            
            for city in os.listdir(img_dir):
                city_img_path = os.path.join(img_dir, city)
                city_gt_path = os.path.join(gt_dir, city)
                
                if not os.path.isdir(city_img_path):
                    continue
                    
                for img_file in tqdm(os.listdir(city_img_path)):
                    if not img_file.endswith('.jpg'):
                        continue
                        
                    # Load image
                    img_path = os.path.join(city_img_path, img_file)
                    image = cv2.imread(img_path)
                    
                    if image is None:
                        continue
                        
                    # Resize image
                    image = cv2.resize(image, self.target_size)
                    image = image.astype(np.float32) / 255.0
                    
                    # Find corresponding GT file
                    gt_file = img_file.replace('leftImg8bit.jpg', 'gtFine_polygons.json')
                    gt_path = os.path.join(city_gt_path, gt_file)
                    
                    # Extract person/rider information for drowsiness context
                    has_person = self.extract_person_info(gt_path)
                    
                    X_data.append(image)
                    y_data.append(1 if has_person else 0)  # Binary: person present or not
        
        # Convert to numpy arrays
        X_data = np.array(X_data)
        y_data = np.eye(2)[y_data]  # One-hot encoding
        
        # Save processed data
        os.makedirs(self.output_path, exist_ok=True)
        np.savez_compressed(
            os.path.join(self.output_path, 'idd_processed_drowsiness.npz'),
            X=X_data, y=y_data
        )
        
        print(f"Processed IDD data: {X_data.shape[0]} samples")
        return X_data, y_data
    
    def extract_person_info(self, gt_path):
        """Extract person/rider information from GT JSON"""
        try:
            with open(gt_path, 'r') as f:
                gt_data = json.load(f)
            
            for obj in gt_data.get('objects', []):
                if obj.get('label') in ['person', 'rider']:
                    return True
            return False
        except:
            return False

if __name__ == "__main__":
    preprocessor = IDDPreprocessor(
        dataset_path="data/raw/IDD_dataset.zip",
        output_path="data/processed"
    )
    preprocessor.process_idd_for_drowsiness()
