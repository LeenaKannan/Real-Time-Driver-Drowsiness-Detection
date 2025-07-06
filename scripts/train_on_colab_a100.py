# scripts/train_colab_a100.py
import os
import numpy as np
import tensorflow as tf
from google.colab import drive, files
import zipfile
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Mount Google Drive
drive.mount('/content/drive')

# Configure A100 GPU with mixed precision
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Found {len(gpus)} GPU(s)")
        print(f"GPU Name: {tf.config.experimental.get_device_details(gpus[0])}")
    except RuntimeError as e:
        print(e)

# Enable mixed precision for A100
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

class A100DrowsinessTrainer:
    def __init__(self):
        self.setup_directories()
        self.model = None
        
    def setup_directories(self):
        """Setup working directories"""
        os.makedirs('/content/data/processed', exist_ok=True)
        os.makedirs('/content/models', exist_ok=True)
        os.makedirs('/content/exports', exist_ok=True)
        
    def upload_and_process_datasets(self):
        """Upload and process datasets"""
        print("Upload your datasets...")
        uploaded = files.upload()
        
        for filename in uploaded.keys():
            if 'IDD_dataset.zip' in filename:
                print("Processing IDD dataset...")
                # Extract and process IDD dataset
                with zipfile.ZipFile(filename, 'r') as zip_ref:
                    zip_ref.extractall('/content/data/raw/')
                    
            elif 'indian_vehicle_dataset.zip' in filename:
                print("Processing Indian vehicle dataset...")
                # Extract and process vehicle dataset
                with zipfile.ZipFile(filename, 'r') as zip_ref:
                    zip_ref.extractall('/content/data/raw/')
    
    def build_optimized_model(self, input_shape=(128, 128, 3), num_classes=2):
        """Build A100-optimized model with mixed precision"""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            
            # Efficient CNN backbone
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
            
            tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.GlobalAveragePooling2D(),
            
            # Efficient head
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(num_classes, activation='softmax', dtype='float32')
        ])
        
        # Compile with mixed precision optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def load_and_combine_datasets(self):
        """Load and combine processed datasets"""
        # Load existing processed data or create dummy data for demo
        try:
            # Try to load processed data
            mrl_data = np.load('/content/data/processed/mrl_processed.npz')
            X_mrl, y_mrl = mrl_data['X'], mrl_data['y']
        except:
            # Create dummy data for demonstration
            print("Creating dummy training data...")
            X_mrl = np.random.random((1000, 128, 128, 3)).astype(np.float32)
            y_mrl = np.random.randint(0, 2, (1000, 2)).astype(np.float32)
            
        return X_mrl, y_mrl
    
    def train_model(self):
        """Train model with A100 optimizations"""
        print("Loading datasets...")
        X_data, y_data = self.load_and_combine_datasets()
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_data, y_data, test_size=0.2, random_state=42
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        
        # Build model
        self.model = self.build_optimized_model()
        print(self.model.summary())
        
        # Data augmentation for better generalization
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            zoom_range=0.1
        )
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=15, restore_best_weights=True, monitor='val_accuracy'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.5, patience=7, min_lr=1e-7
            ),
            tf.keras.callbacks.ModelCheckpoint(
                '/content/models/best_drowsiness_model.keras',
                save_best_only=True,
                monitor='val_accuracy',
                mode='max'
            )
        ]
        
        # Train with high batch size for A100
        history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=128),
            validation_data=(X_val, y_val),
            epochs=100,
            callbacks=callbacks,
            verbose=1,
            steps_per_epoch=len(X_train) // 128
        )
        
        # Save final model
        self.model.save('/content/models/drowsiness_detection_final.keras')
        
        # Plot training history
        self.plot_training_history(history)
        
        # Save to Drive
        self.model.save('/content/drive/MyDrive/drowsiness_model_a100.keras')
        
        return self.model, history
    
    def plot_training_history(self, history):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('/content/training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

# Run training
if __name__ == "__main__":
    trainer = A100DrowsinessTrainer()
    trainer.upload_and_process_datasets()
    model, history = trainer.train_model()
    print("Training completed! Model saved to Google Drive.")
