# scripts/export_model.py
import tensorflow as tf
import numpy as np
import os

class ModelExporter:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = tf.keras.models.load_model(model_path)
        
    def export_to_tflite(self, output_path, quantize=True):
        """Export model to TensorFlow Lite format"""
        print("Converting to TensorFlow Lite...")
        
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        if quantize:
            # Apply quantization for smaller model size
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            # Representative dataset for better quantization
            def representative_data_gen():
                for _ in range(100):
                    yield [np.random.random((1, 128, 128, 3)).astype(np.float32)]
            
            converter.representative_dataset = representative_data_gen
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
        
        tflite_model = converter.convert()
        
        # Save the model
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        # Print size comparison
        original_size = os.path.getsize(self.model_path) / (1024 * 1024)
        tflite_size = os.path.getsize(output_path) / (1024 * 1024)
        
        print(f"Original model size: {original_size:.2f} MB")
        print(f"TFLite model size: {tflite_size:.2f} MB")
        print(f"Size reduction: {((original_size - tflite_size) / original_size * 100):.1f}%")
        
        return output_path
    
    def export_to_onnx(self, output_path):
        """Export model to ONNX format"""
        try:
            import tf2onnx
            
            print("Converting to ONNX...")
            
            # Convert to ONNX
            onnx_model, _ = tf2onnx.convert.from_keras(
                self.model,
                input_signature=None,
                opset=13,
                output_path=output_path
            )
            
            print(f"ONNX model saved to: {output_path}")
            return output_path
            
        except ImportError:
            print("tf2onnx not installed. Install with: pip install tf2onnx")
            return None
    
    def test_tflite_model(self, tflite_path):
        """Test TFLite model inference"""
        print("Testing TFLite model...")
        
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"Input shape: {input_details[0]['shape']}")
        print(f"Input type: {input_details[0]['dtype']}")
        print(f"Output shape: {output_details[0]['shape']}")
        
        # Test with random input
        if input_details[0]['dtype'] == np.uint8:
            input_data = np.random.randint(0, 255, input_details[0]['shape'], dtype=np.uint8)
        else:
            input_data = np.random.random(input_details[0]['shape']).astype(np.float32)
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        output_data = interpreter.get_tensor(output_details[0]['index'])
        print(f"Test output: {output_data}")
        print("TFLite model test successful!")

if __name__ == "__main__":
    # Export trained model
    exporter = ModelExporter("models/drowsiness_detection_final.keras")
    
    # Export to TFLite (quantized)
    tflite_path = exporter.export_to_tflite(
        "models/drowsiness_detection_quantized.tflite",
        quantize=True
    )
    
    # Export to ONNX
    onnx_path = exporter.export_to_onnx("models/drowsiness_detection.onnx")
    
    # Test TFLite model
    exporter.test_tflite_model(tflite_path)
