# scripts/train_colab_a100.py
import os
import numpy as np
import tensorflow as tf
from google.colab import drive, files
import zipfile
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import sys
import gc

# Mount Google Drive
drive.mount('/content/drive')

# Configure A100 GPU with optimal settings
def setup_gpu():
    """Setup A100 GPU with optimal configurations"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth to avoid OOM
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Set memory limit to 38GB (leaving some for system)
            tf.config.experimental.set_memory_limit(gpus[0], 38000)
            
            print(f"Found {len(gpus)} GPU(s)")
            print(f"GPU Details: {tf.config.experimental.get_device_details(gpus[0])}")
            
            # Enable mixed precision for A100
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print("Mixed precision enabled for A100")
            
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    else:
        print("No GPU found!")

class A100DrowsinessTrainer:
    def __init__(self):
        self.setup_directories()
        self.model = None
        self.history = None
        
    def setup_directories(self):
        """Setup working directories"""
        directories = [
            '/content/data/raw/MRL_dataset',
            '/content/data/raw/YawDD_dataset',
            '/content/data/processed',
            '/content/models',
            '/content/exports',
            '/content/logs'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            
    def upload_datasets(self):
        """Upload and extract datasets"""
        print("📁 Upload your datasets (MRL and YawDD)...")
        uploaded = files.upload()
        
        for filename in uploaded.keys():
            print(f"Processing {filename}...")
            
            if filename.endswith('.zip'):
                with zipfile.ZipFile(filename, 'r') as zip_ref:
                    if 'mrl' in filename.lower():
                        zip_ref.extractall('/content/data/raw/MRL_dataset/')
                        print("✅ MRL dataset extracted")
                    elif 'yawdd' in filename.lower():
                        zip_ref.extractall('/content/data/raw/YawDD_dataset/')
                        print("✅ YawDD dataset extracted")
                    else:
                        zip_ref.extractall('/content/data/raw/')
                        print(f"✅ {filename} extracted")
                        
    def load_and_preprocess_datasets(self):
        """Load and preprocess both MRL and YawDD datasets"""
        print("🔄 Loading and preprocessing datasets...")
        
        # Import data preprocessing
        sys.path.append('/content')
        from src.models.data_preprocessing import DataPreprocessor
        
        preprocessor = DataPreprocessor(target_size=(128, 128))
        
        # Load MRL dataset (Open/Closed eyes)
        try:
            print("Loading MRL dataset...")
            X_mrl, y_mrl = preprocessor.load_mrl_dataset('/content/data/raw/MRL_dataset')
            print(f"MRL dataset loaded: {X_mrl.shape[0]} samples")
            
            # Convert to binary classification (0: Open/Awake, 1: Closed/Drowsy)
            y_mrl_binary = np.argmax(y_mrl, axis=1)
            
        except Exception as e:
            print(f"MRL dataset loading failed: {e}")
            X_mrl, y_mrl_binary = None, None
            
        # Load YawDD dataset
        try:
            print("Loading YawDD dataset...")
            X_yawdd, y_yawdd = preprocessor.load_yawdd_frames('/content/data/raw/YawDD_dataset')
            print(f"YawDD dataset loaded: {X_yawdd.shape[0]} samples")
            
            # Convert YawDD to binary (0: Open/No_Yawn, 1: Close/Yawn)
            y_yawdd_labels = np.argmax(y_yawdd, axis=1)
            y_yawdd_binary = np.where((y_yawdd_labels == 1) | (y_yawdd_labels == 2), 1, 0)
            
        except Exception as e:
            print(f"YawDD dataset loading failed: {e}")
            X_yawdd, y_yawdd_binary = None, None
            
        # Combine datasets
        X_combined = []
        y_combined = []
        
        if X_mrl is not None:
            X_combined.append(X_mrl)
            y_combined.append(y_mrl_binary)
            
        if X_yawdd is not None:
            X_combined.append(X_yawdd)
            y_combined.append(y_yawdd_binary)
            
        if X_combined:
            X_final = np.vstack(X_combined)
            y_final = np.hstack(y_combined)
            
            # Convert to categorical
            y_final = tf.keras.utils.to_categorical(y_final, 2)
            
            print(f"Combined dataset: {X_final.shape[0]} samples")
            print(f"Class distribution: {np.bincount(np.argmax(y_final, axis=1))}")
            
            # Save processed data
            np.savez_compressed('/content/data/processed/combined_dataset.npz',
                              X=X_final, y=y_final)
            
            return X_final, y_final
        else:
            raise ValueError("No datasets could be loaded!")
            
    def build_optimized_model(self, input_shape=(128, 128, 3)):
        """Build A100-optimized model with mixed precision"""
        model = tf.keras.Sequential([
            # Input layer
            tf.keras.layers.Input(shape=input_shape),
            
            # Data augmentation layer
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomBrightness(0.1),
            
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
            
            # Classification head
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(2, activation='softmax', dtype='float32')
        ])
        
        # Compile with mixed precision optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall', 'f1_score']
        )
        
        return model
        
    def train_model(self, X_data, y_data):
        """Train model with A100 optimizations"""
        print("🚀 Starting training on A100 GPU...")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_data, y_data, test_size=0.2, random_state=42, stratify=np.argmax(y_data, axis=1)
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        
        # Calculate class weights for imbalanced data
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(np.argmax(y_train, axis=1)),
            y=np.argmax(y_train, axis=1)
        )
        class_weight_dict = dict(enumerate(class_weights))
        print(f"Class weights: {class_weight_dict}")
        
        # Build model
        self.model = self.build_optimized_model()
        print(self.model.summary())
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=20, restore_best_weights=True, monitor='val_accuracy'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.5, patience=10, min_lr=1e-7, verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                '/content/models/best_drowsiness_model.keras',
                save_best_only=True,
                monitor='val_accuracy',
                mode='max',
                verbose=1
            ),
            tf.keras.callbacks.CSVLogger('/content/logs/training.log')
        ]
        
        # Train with large batch size for A100
        batch_size = 256  # A100 can handle large batches
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=150,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1
        )
        
        # Save final model
        self.model.save('/content/models/drowsiness_detection_final.keras')
        
        # Save to Drive
        self.model.save('/content/drive/MyDrive/drowsiness_model_a100.keras')
        
        # Plot training history
        self.plot_training_history()
        
        # Evaluate model
        self.evaluate_model(X_val, y_val)
        
        return self.model, self.history
        
    def plot_training_history(self):
        """Plot comprehensive training history"""
        if self.history is None:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision
        axes[1, 0].plot(self.history.history['precision'], label='Training Precision')
        axes[1, 0].plot(self.history.history['val_precision'], label='Validation Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Recall
        axes[1, 1].plot(self.history.history['recall'], label='Training Recall')
        axes[1, 1].plot(self.history.history['val_recall'], label='Validation Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('/content/training_history.png', dpi=300, bbox_inches='tight')
        plt.savefig('/content/drive/MyDrive/training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def evaluate_model(self, X_val, y_val):
        """Comprehensive model evaluation"""
        print("📊 Evaluating model...")
        
        # Predictions
        y_pred = self.model.predict(X_val)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_val, axis=1)
        
        # Classification report
        from sklearn.metrics import classification_report, confusion_matrix
        
        print("\nClassification Report:")
        print(classification_report(y_true_classes, y_pred_classes, 
                                  target_names=['Awake', 'Drowsy']))
        
        # Confusion matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ['Awake', 'Drowsy'])
        plt.yticks(tick_marks, ['Awake', 'Drowsy'])
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('/content/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.savefig('/content/drive/MyDrive/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def export_optimized_model(self):
        """Export model for Raspberry Pi deployment"""
        print("📦 Exporting optimized model for Raspberry Pi...")
        
        if self.model is None:
            print("No model to export!")
            return
            
        # Convert to TensorFlow Lite with quantization
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        # Apply optimizations
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Representative dataset for quantization
        def representative_data_gen():
            for _ in range(100):
                yield [np.random.random((1, 128, 128, 3)).astype(np.float32)]
        
        converter.representative_dataset = representative_data_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        
        # Convert
        tflite_model = converter.convert()
        
        # Save optimized model
        tflite_path = '/content/models/drowsiness_detection_quantized.tflite'
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
            
        # Save to Drive
        drive_path = '/content/drive/MyDrive/drowsiness_detection_quantized.tflite'
        with open(drive_path, 'wb') as f:
            f.write(tflite_model)
        
        # Size comparison
        original_size = len(tf.keras.models.model_to_json(self.model).encode('utf-8')) / (1024 * 1024)
        tflite_size = len(tflite_model) / (1024 * 1024)
        
        print(f"Original model size: ~{original_size:.2f} MB")
        print(f"TFLite model size: {tflite_size:.2f} MB")
        print(f"Size reduction: {((original_size - tflite_size) / original_size * 100):.1f}%")
        
        return tflite_path

def main():
    """Main training pipeline"""
    print("🚀 Starting A100 GPU Training Pipeline")
    print("=" * 50)
    
    # Setup GPU
    setup_gpu()
    
    # Initialize trainer
    trainer = A100DrowsinessTrainer()
    
    # Upload datasets
    trainer.upload_datasets()
    
    # Load and preprocess data
    X_data, y_data = trainer.load_and_preprocess_datasets()
    
    # Train model
    model, history = trainer.train_model(X_data, y_data)
    
    # Export optimized model
    trainer.export_optimized_model()
    
    print("✅ Training completed successfully!")
    print("📁 Files saved to Google Drive:")
    print("  - drowsiness_model_a100.keras")
    print("  - drowsiness_detection_quantized.tflite")
    print("  - training_history.png")
    print("  - confusion_matrix.png")

if __name__ == "__main__":
    main()
