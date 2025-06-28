# src/models/train_model.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from model_architecture import DrowsinessDetectionModel

class ModelTrainer:
    def __init__(self, dataset_path, model_type='cnn'):
        self.dataset_path = dataset_path
        self.model_type = model_type
        self.model_builder = DrowsinessDetectionModel()
        
    def load_yawdd_dataset(self):
        """Load and preprocess YawDD dataset"""
        # YawDD dataset structure: Normal/Yawn/Talk
        X, y = [], []
        
        # Process normal driving images
        normal_path = f"{self.dataset_path}/Normal"
        for img_file in os.listdir(normal_path):
            img = cv2.imread(os.path.join(normal_path, img_file))
            img = cv2.resize(img, (128, 128))
            img = img / 255.0
            X.append(img)
            y.append([1, 0])  # Awake
            
        # Process yawning images
        yawn_path = f"{self.dataset_path}/Yawn"
        for img_file in os.listdir(yawn_path):
            img = cv2.imread(os.path.join(yawn_path, img_file))
            img = cv2.resize(img, (128, 128))
            img = img / 255.0
            X.append(img)
            y.append([0, 1])  # Drowsy
            
        return np.array(X), np.array(y)
    
    def load_mrl_dataset(self):
        """Load MRL Eye Dataset for eye state classification"""
        # MRL dataset has 84,898 samples with high accuracy potential
        X, y = [], []
        
        # Process open eyes
        open_eyes_path = f"{self.dataset_path}/Open_Eyes"
        for img_file in os.listdir(open_eyes_path):
            img = cv2.imread(os.path.join(open_eyes_path, img_file))
            img = cv2.resize(img, (128, 128))
            img = img / 255.0
            X.append(img)
            y.append([1, 0])  # Open
            
        # Process closed eyes
        closed_eyes_path = f"{self.dataset_path}/Closed_Eyes"
        for img_file in os.listdir(closed_eyes_path):
            img = cv2.imread(os.path.join(closed_eyes_path, img_file))
            img = cv2.resize(img, (128, 128))
            img = img / 255.0
            X.append(img)
            y.append([0, 1])  # Closed
            
        return np.array(X), np.array(y)
    
    def train_model(self):
        """Train the drowsiness detection model"""
        # Load dataset
        if 'yawdd' in self.dataset_path.lower():
            X, y = self.load_yawdd_dataset()
        else:
            X, y = self.load_mrl_dataset()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Data augmentation
        datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1
        )
        
        # Build model
        if self.model_type == 'cnn':
            model = self.model_builder.build_cnn_model()
        else:
            model = self.model_builder.build_vit_model()
        
        # Training callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
            tf.keras.callbacks.ModelCheckpoint('models/best_model.h5', save_best_only=True)
        ]
        
        # Train model
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=32),
            epochs=100,
            validation_data=(X_test, y_test),
            callbacks=callbacks
        )
        
        # Evaluate model
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Accuracy: {test_acc:.4f}")
        
        # Save final model
        model.save('models/drowsiness_detection_model.h5')
        
        return model, history

if __name__ == "__main__":
    trainer = ModelTrainer('data/YawDD', model_type='cnn')
    model, history = trainer.train_model()
