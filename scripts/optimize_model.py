# scripts/optimize_model.py
import tensorflow as tf
import numpy as np
import argparse

def optimize_model_for_pi(model_path, output_path):
    """Optimize model for Raspberry Pi deployment"""
    print(f"Loading model from {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    # Convert to TensorFlow Lite with optimizations
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
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
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    # Print size comparison
    import os
    original_size = os.path.getsize(model_path) / (1024 * 1024)
    optimized_size = os.path.getsize(output_path) / (1024 * 1024)
    
    print(f"Original model size: {original_size:.2f} MB")
    print(f"Optimized model size: {optimized_size:.2f} MB")
    print(f"Size reduction: {((original_size - optimized_size) / original_size * 100):.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input model path')
    parser.add_argument('--output', required=True, help='Output TFLite model path')
    
    args = parser.parse_args()
    optimize_model_for_pi(args.input, args.output)
