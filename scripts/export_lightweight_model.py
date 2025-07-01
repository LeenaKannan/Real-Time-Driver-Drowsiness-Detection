# scripts/export_lightweight_model.py
import tensorflow as tf
import argparse
import os

def convert_to_tflite(model_path, output_path, quantize=True):
    """Convert Keras model to TensorFlow Lite for Raspberry Pi"""
    model = tf.keras.models.load_model(model_path)
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # Optional: Use representative dataset for better quantization
        converter.representative_dataset = representative_data_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
    
    tflite_model = converter.convert()
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"Model converted and saved to {output_path}")
    print(f"Original size: {os.path.getsize(model_path) / 1024 / 1024:.2f} MB")
    print(f"TFLite size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

def representative_data_gen():
    """Generate representative data for quantization"""
    import numpy as np
    for _ in range(100):
        yield [np.random.random((1, 128, 128, 3)).astype(np.float32)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, help='Path to Keras model')
    parser.add_argument('--output_path', required=True, help='Output TFLite model path')
    parser.add_argument('--quantize', action='store_true', help='Apply quantization')
    
    args = parser.parse_args()
    convert_to_tflite(args.model_path, args.output_path, args.quantize)
