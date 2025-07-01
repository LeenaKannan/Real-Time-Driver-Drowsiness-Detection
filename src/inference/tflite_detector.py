# src/inference/tflite_detector.py
import tensorflow as tf
import numpy as np
import cv2
import mediapipe as mp
from collections import deque

class TFLiteDrowsinessDetector:
    def __init__(self, model_path, ear_threshold=0.25, yawn_threshold=0.6):
        # Load TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        self.ear_threshold = ear_threshold
        self.yawn_threshold = yawn_threshold
        
        # MediaPipe setup
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Eye landmarks for MediaPipe
        self.LEFT_EYE_LANDMARKS = [33, 7, 163, 144, 145, 153]
        self.RIGHT_EYE_LANDMARKS = [362, 382, 381, 380, 374, 373]
        self.MOUTH_LANDMARKS = [78, 95, 88, 178, 87, 14, 317, 402, 318]
        
        # Tracking
        self.drowsy_frames = 0
        self.ear_history = deque(maxlen=30)
    
    def preprocess_eye_region(self, eye_region):
        """Preprocess eye region for TFLite model"""
        if eye_region is None or eye_region.size == 0:
            return None
        
        # Resize to model input size
        eye_region = cv2.resize(eye_region, (128, 128))
        eye_region = eye_region.astype(np.float32) / 255.0
        eye_region = np.expand_dims(eye_region, axis=0)
        
        return eye_region
    
    def predict_eye_state(self, eye_region):
        """Predict eye state using TFLite model"""
        preprocessed = self.preprocess_eye_region(eye_region)
        if preprocessed is None:
            return 0.5
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], preprocessed)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output_data[0][1]  # Probability of closed eye
    
    def extract_eye_region(self, frame, landmarks, eye_indices):
        """Extract eye region from frame"""
        h, w = frame.shape[:2]
        
        eye_points = []
        for idx in eye_indices:
            x = int(landmarks.landmark[idx].x * w)
            y = int(landmarks.landmark[idx].y * h)
            eye_points.append([x, y])
        
        if not eye_points:
            return None
        
        eye_points = np.array(eye_points)
        x_min, y_min = np.min(eye_points, axis=0)
        x_max, y_max = np.max(eye_points, axis=0)
        
        # Add padding
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        
        eye_region = frame[y_min:y_max, x_min:x_max]
        return eye_region if eye_region.size > 0 else None
    
    def detect_drowsiness(self, frame):
        """Main drowsiness detection function optimized for Pi"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return {
                'status': 'NO_FACE',
                'confidence': 0.0,
                'ear': 0.0,
                'mar': 0.0,
                'alert_required': False
            }
        
        landmarks = results.multi_face_landmarks[0]
        
        # Extract eye regions
        left_eye = self.extract_eye_region(frame, landmarks, self.LEFT_EYE_LANDMARKS)
        right_eye = self.extract_eye_region(frame, landmarks, self.RIGHT_EYE_LANDMARKS)
        
        # Predict eye states
        left_closed_prob = self.predict_eye_state(left_eye)
        right_closed_prob = self.predict_eye_state(right_eye)
        avg_closed_prob = (left_closed_prob + right_closed_prob) / 2
        
        # Simple EAR calculation as backup
        ear = self.calculate_simple_ear(landmarks)
        self.ear_history.append(ear)
        
        # Drowsiness logic
        is_drowsy = False
        status = "AWAKE"
        confidence = 0.0
        
        if avg_closed_prob > 0.7 or ear < self.ear_threshold:
            self.drowsy_frames += 1
            if self.drowsy_frames >= 15:  # ~0.5 seconds at 30 FPS
                is_drowsy = True
                status = "DROWSY"
                confidence = max(avg_closed_prob, 1 - ear)
        else:
            self.drowsy_frames = max(0, self.drowsy_frames - 2)
        
        if self.drowsy_frames >= 45:  # 1.5 seconds
            status = "SLEEPING"
            confidence = 1.0
        
        return {
            'status': status,
            'confidence': confidence,
            'ear': ear,
            'mar': 0.0,  # Simplified for Pi
            'alert_required': is_drowsy
        }
    
    def calculate_simple_ear(self, landmarks):
        """Simplified EAR calculation for Pi"""
        try:
            # Get key eye points
            left_eye = [(landmarks.landmark[i].x, landmarks.landmark[i].y) 
                       for i in self.LEFT_EYE_LANDMARKS]
            right_eye = [(landmarks.landmark[i].x, landmarks.landmark[i].y) 
                        for i in self.RIGHT_EYE_LANDMARKS]
            
            # Simple vertical/horizontal ratio
            left_ear = abs(left_eye[1][1] - left_eye[4][1]) / abs(left_eye[0][0] - left_eye[3][0])
            right_ear = abs(right_eye[1][1] - right_eye[4][1]) / abs(right_eye[0][0] - right_eye[3][0])
            
            return (left_ear + right_ear) / 2
        except:
            return 0.3  # Default value
