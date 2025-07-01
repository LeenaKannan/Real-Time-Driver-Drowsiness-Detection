# scripts/train_on_colab.py
import os
import numpy as np
import tensorflow as tf
from google.colab import drive, files
import zipfile
import requests

# Mount Google Drive
drive.mount('/content/drive')

# Configure A100 GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Found {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(e)

# Mixed precision for A100
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

class ColabTrainer:
    def __init__(self):
        self.setup_directories()
        self.download_datasets()
    
    def setup_directories(self):
        """Setup working directories"""
        os.makedirs('/content/data/raw', exist_ok=True)
        os.makedirs('/content/data/processed', exist_ok=True)
        os.makedirs('/content/models', exist_ok=True)
    
    def download_datasets(self):
        """Download and extract datasets"""
        # MRL Dataset
        if not os.path.exists('/content/data/raw/MRL_dataset'):
            print("Downloading MRL Eye Dataset...")
            os.system('wget -O /content/mrl_dataset.zip http://mrl.cs.vsb.cz/eyedataset/mrl_dataset.zip')
            with zipfile.ZipFile('/content/mrl_dataset.zip', 'r') as zip_ref:
                zip_ref.extractall('/content/data/raw/')
        
        # YawDD Dataset (if available)
        print("Datasets ready for processing")
    
    def build_optimized_model(self, input_shape=(128, 128, 3)):
        """Build A100-optimized model with mixed precision"""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            
            # Efficient CNN blocks
            tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.Dropout(0.25),
            
            tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.Dropout(0.25),
            
            tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.Dropout(0.25),
            
            # Efficient head
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(2, activation='softmax', dtype='float32')  # Keep output as float32
        ])
        
        # Compile with mixed precision
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self):
        """Train model with A100 optimizations"""
        # Load processed data
        train_data = np.load('/content/data/processed/mrl_processed.npz')
        X_train, y_train = train_data['X'], train_data['y']
        
        # Split data
        split_idx = int(0.8 * len(X_train))
        X_val, y_val = X_train[split_idx:], y_train[split_idx:]
        X_train, y_train = X_train[:split_idx], y_train[:split_idx]
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        
        # Build model
        model = self.build_optimized_model()
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
            tf.keras.callbacks.ModelCheckpoint(
                '/content/models/best_model.keras',
                save_best_only=True,
                monitor='val_accuracy'
            )
        ]
        
        # Train with high batch size for A100
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=128,  # Large batch for A100
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        model.save('/content/models/drowsiness_detection_model.keras')
        
        # Convert to TFLite for Pi
        self.convert_for_pi(model)
        
        # Save to Drive
        model.save('/content/drive/MyDrive/drowsiness_model.keras')
        
        return model, history
    
    def convert_for_pi(self, model):
        """Convert model for Raspberry Pi deployment"""
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()
        
        with open('/content/models/drowsiness_model.tflite', 'wb') as f:
            f.write(tflite_model)
        
        # Save to Drive
        files.download('/content/models/drowsiness_model.tflite')
        
        print("TFLite model ready for Raspberry Pi!")

# Run training
if __name__ == "__main__":
    trainer = ColabTrainer()
    model, history = trainer.train_model()
