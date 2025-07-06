# src/inference/pi_camera_inference.py
import cv2
import numpy as np
import tensorflow as tf
import time
import threading
from queue import Queue
import mediapipe as mp
from picamera2 import Picamera2
from collections import deque

class PiCameraInference:
    def __init__(self, model_path, camera_resolution=(640, 480)):
        self.model_path = model_path
        self.camera_resolution = camera_resolution
        self.frame_queue = Queue(maxsize=2)
        self.result_queue = Queue(maxsize=5)
        
        # Initialize TFLite interpreter
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Initialize MediaPipe for face detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.7
        )
        
        # Initialize camera
        self.camera = Picamera2()
        self.setup_camera()
        
        # Tracking variables
        self.drowsy_frames = 0
        self.alert_history = deque(maxlen=30)  # 1 second at 30 FPS
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
    def setup_camera(self):
        """Setup Pi camera with optimal settings"""
        config = self.camera.create_preview_configuration(
            main={"size": self.camera_resolution, "format": "RGB888"}
        )
        self.camera.configure(config)
        self.camera.start()
        time.sleep(2)  # Camera warm-up
        
    def preprocess_frame(self, frame):
        """Preprocess frame for model inference"""
        # Extract face region if possible
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        
        if results.detections:
            # Use first detected face
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            
            h, w = frame.shape[:2]
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            
            # Extract face region with padding
            padding = 20
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(w, x + width + padding)
            y2 = min(h, y + height + padding)
            
            face_region = frame[y1:y2, x1:x2]
            
            if face_region.size > 0:
                face_region = cv2.resize(face_region, (128, 128))
                return face_region
        
        # If no face detected, use center crop
        h, w = frame.shape[:2]
        center_crop = frame[h//4:3*h//4, w//4:3*w//4]
        center_crop = cv2.resize(center_crop, (128, 128))
        return center_crop
    
    def predict_drowsiness(self, frame):
        """Predict drowsiness from frame"""
        processed_frame = self.preprocess_frame(frame)
        
        # Normalize based on model input type
        if self.input_details[0]['dtype'] == np.uint8:
            input_data = processed_frame.astype(np.uint8)
        else:
            input_data = (processed_frame / 255.0).astype(np.float32)
        
        input_data = np.expand_dims(input_data, axis=0)
        
        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        # Get prediction
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # Assuming binary classification: [awake, drowsy]
        drowsy_probability = output_data[0][1]
        is_drowsy = drowsy_probability > 0.7
        
        return is_drowsy, drowsy_probability
    
    def calculate_fps(self):
        """Calculate current FPS"""
        self.fps_counter += 1
        if self.fps_counter % 30 == 0:
            self.current_fps = 30 / (time.time() - self.fps_start_time)
            self.fps_start_time = time.time()
        return self.current_fps
    
    def camera_thread(self):
        """Camera capture thread"""
        while True:
            try:
                frame = self.camera.capture_array()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
                    
            except Exception as e:
                print(f"Camera error: {e}")
                break
    
    def inference_thread(self):
        """Inference processing thread"""
        while True:
            try:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get()
                    
                    # Predict drowsiness
                    is_drowsy, confidence = self.predict_drowsiness(frame)
                    
                    # Update tracking
                    self.alert_history.append(is_drowsy)
                    
                    # Determine alert level
                    recent_drowsy = sum(self.alert_history[-10:])  # Last 10 frames
                    
                    if recent_drowsy >= 7:  # 70% of recent frames
                        alert_level = "CRITICAL"
                    elif recent_drowsy >= 4:  # 40% of recent frames
                        alert_level = "WARNING"
                    else:
                        alert_level = "NORMAL"
                    
                    result = {
                        'frame': frame,
                        'is_drowsy': is_drowsy,
                        'confidence': confidence,
                        'alert_level': alert_level,
                        'fps': self.calculate_fps()
                    }
                    
                    if not self.result_queue.full():
                        self.result_queue.put(result)
                        
            except Exception as e:
                print(f"Inference error: {e}")
                time.sleep(0.1)
    
    def start_inference(self):
        """Start real-time inference"""
        print("Starting real-time drowsiness detection...")
        
        # Start threads
        camera_thread = threading.Thread(target=self.camera_thread, daemon=True)
        inference_thread = threading.Thread(target=self.inference_thread, daemon=True)
        
        camera_thread.start()
        inference_thread.start()
        
        return True
    
    def get_latest_result(self):
        """Get latest inference result"""
        if not self.result_queue.empty():
            return self.result_queue.get()
        return None
    
    def cleanup(self):
        """Cleanup resources"""
        self.camera.stop()
        self.camera.close()

if __name__ == "__main__":
    # Test the inference system
    inference = PiCameraInference("models/drowsiness_detection_quantized.tflite")
    inference.start_inference()
    
    try:
        while True:
            result = inference.get_latest_result()
            if result:
                print(f"Alert: {result['alert_level']}, "
                      f"Confidence: {result['confidence']:.3f}, "
                      f"FPS: {result['fps']:.1f}")
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("Stopping inference...")
        inference.cleanup()
