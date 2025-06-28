# src/models/model_architecture.py
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

class DrowsinessDetectionModel:
    def __init__(self, input_shape=(128, 128, 3)):
        self.input_shape = input_shape
        self.model = None
        
    def build_cnn_model(self):
        """Build CNN model for eye state classification"""
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(2, activation='softmax')  # Open/Closed eyes
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self.model = model
        return model
    
    def build_vit_model(self):
        """Build Vision Transformer for enhanced accuracy"""
        # Simplified ViT implementation
        inputs = layers.Input(shape=self.input_shape)
        
        # Patch extraction
        patches = self.extract_patches(inputs, patch_size=16)
        
        # Transformer encoder
        encoded_patches = self.transformer_encoder(patches, num_heads=8, ff_dim=256)
        
        # Classification head
        representation = layers.GlobalAveragePooling1D()(encoded_patches)
        representation = layers.Dropout(0.3)(representation)
        outputs = layers.Dense(2, activation='softmax')(representation)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def extract_patches(self, images, patch_size):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, patch_size, patch_size, 1],
            strides=[1, patch_size, patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
    
    def transformer_encoder(self, inputs, num_heads, ff_dim):
        # Multi-head attention
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=inputs.shape[-1]
        )(inputs, inputs)
        
        # Skip connection and layer norm
        x1 = layers.Add()([attention_output, inputs])
        x1 = layers.LayerNormalization(epsilon=1e-6)(x1)
        
        # Feed forward network
        ffn_output = layers.Dense(ff_dim, activation='relu')(x1)
        ffn_output = layers.Dense(inputs.shape[-1])(ffn_output)
        
        # Skip connection and layer norm
        x2 = layers.Add()([ffn_output, x1])
        return layers.LayerNormalization(epsilon=1e-6)(x2)
