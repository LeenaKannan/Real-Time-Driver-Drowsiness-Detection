# src/inference/drowsiness_detector.py
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from collections import deque
import time

class DrowsinessDetector:
    def __init__(self, model_path, model_type='cnn'):
        self.model_path = model_path
        self.model_type = model_type
        
        # Load model
        if model_path.endswith('.tflite'):
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.use_tflite = True
        else:
            self.model = tf.keras.models.load_model(model_path)
            self.use_tflite = False
        
        # Initialize MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Eye landmark indices for EAR calculation
        self.LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        # Mouth landmark indices for MAR calculation
        self.MOUTH_INDICES = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
        
        # Tracking variables
        self.ear_history = deque(maxlen=30)
        self.mar_history = deque(maxlen=30)
        self.drowsy_frames = 0
        self.yawn_frames = 0
        
        # Thresholds
        self.EAR_THRESHOLD = 0.25
        self.MAR_THRESHOLD = 0.6
        self.DROWSY_FRAME_THRESHOLD = 20
        self.YAWN_FRAME_THRESHOLD = 15
    
    def calculate_ear(self, landmarks, eye_indices):
        """Calculate Eye Aspect Ratio"""
        eye_points = np.array([[landmarks[i].x, landmarks[i].y] for i in eye_indices])
        
        # Calculate distances
        A = np.linalg.norm(eye_points[1] - eye_points[5])
        B = np.linalg.norm(eye_points[2] - eye_points[4])
        C = np.linalg.norm(eye_points[0] - eye_points[3])
        
        ear = (A + B) / (2.0 * C)
        return ear
    
    def calculate_mar(self, landmarks, mouth_indices):
        """Calculate Mouth Aspect Ratio"""
        mouth_points = np.array([[landmarks[i].x, landmarks[i].y] for i in mouth_indices[:8]])
        
        # Calculate vertical distances
        A = np.linalg.norm(mouth_points[1] - mouth_points[7])
        B = np.linalg.norm(mouth_points[2] - mouth_points[6])
        C = np.linalg.norm(mouth_points[3] - mouth_points[5])
        
        # Calculate horizontal distance
        D = np.linalg.norm(mouth_points[0] - mouth_points[4])
        
        mar = (A + B + C) / (3.0 * D)
        return mar
    
    def extract_face_region(self, frame, landmarks):
        """Extract face region for model prediction"""
        h, w = frame.shape[:2]
        
        # Get face bounding box
        x_coords = [landmark.x * w for landmark in landmarks]
        y_coords = [landmark.y * h for landmark in landmarks]
        
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        
        # Add padding
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        
        face_region = frame[y_min:y_max, x_min:x_max]
        
        if face_region.size > 0:
            face_region = cv2.resize(face_region, (128, 128))
            return face_region
        
        return None
    
    def predict_drowsiness_ml(self, face_region):
        """Predict drowsiness using ML model"""
        if face_region is None:
            return False, 0.0
        
        # Preprocess
        if self.use_tflite:
            if self.input_details[0]['dtype'] == np.uint8:
                input_data = face_region.astype(np.uint8)
            else:
                input_data = (face_region / 255.0).astype(np.float32)
            
            input_data = np.expand_dims(input_data, axis=0)
            
            # Run inference
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            drowsy_prob = output_data[0][1]  # Assuming [awake, drowsy]
        else:
            input_data = np.expand_dims(face_region / 255.0, axis=0)
            prediction = self.model.predict(input_data, verbose=0)
            drowsy_prob = prediction[0][1]
        
        is_drowsy = drowsy_prob > 0.7
        return is_drowsy, drowsy_prob
    
    def detect_drowsiness(self, frame):
        """Main drowsiness detection function"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return {
                'is_drowsy': False,
                'status': 'NO_FACE',
                'confidence': 0.0,
                'ear': 0.0,
                'mar': 0.0
            }
        
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Calculate EAR and MAR
        left_ear = self.calculate_ear(landmarks, self.LEFT_EYE_INDICES)
        right_ear = self.calculate_ear(landmarks, self.RIGHT_EYE_INDICES)
        avg_ear = (left_ear + right_ear) / 2.0
        
        mar = self.calculate_mar(landmarks, self.MOUTH_INDICES)
        
        # Update history
        self.ear_history.append(avg_ear)
        self.mar_history.append(mar)
        
        # Detect drowsiness based on EAR
        if avg_ear < self.EAR_THRESHOLD:
            self.drowsy_frames += 1
        else:
            self.drowsy_frames = max(0, self.drowsy_frames - 1)
        
        # Detect yawning based on MAR
        if mar > self.MAR_THRESHOLD:
            self.yawn_frames += 1
        else:
            self.yawn_frames = max(0, self.yawn_frames - 1)
        
        # ML-based prediction
        face_region = self.extract_face_region(frame, landmarks)
        ml_drowsy, ml_confidence = self.predict_drowsiness_ml(face_region)
        
        # Combine rule-based and ML predictions
        is_drowsy = False
        status = 'AWAKE'
        confidence = ml_confidence
        
        if self.yawn_frames > self.YAWN_FRAME_THRESHOLD:
            is_drowsy = True
            status = 'YAWNING'
            confidence = max(confidence, 0.8)
        elif self.drowsy_frames > self.DROWSY_FRAME_THRESHOLD:
            is_drowsy = True
            status = 'SLEEPING'
            confidence = max(confidence, 0.9)
        elif ml_drowsy or avg_ear < self.EAR_THRESHOLD:
            is_drowsy = True
            status = 'DROWSY'
        
        return {
            'is_drowsy': is_drowsy,
            'status': status,
            'confidence': confidence,
            'ear': avg_ear,
            'mar': mar
        }
