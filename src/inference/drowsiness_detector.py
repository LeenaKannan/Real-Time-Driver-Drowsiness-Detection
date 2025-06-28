# src/inference/drowsiness_detector.py
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from scipy.spatial import distance as dist
import time
import threading
from collections import deque

class DrowsinessDetector:
    def __init__(self, model_path, ear_threshold=0.25, yawn_threshold=0.6):
        self.model = tf.keras.models.load_model(model_path)
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
        
        # Eye landmarks indices for MediaPipe
        self.LEFT_EYE_LANDMARKS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE_LANDMARKS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.MOUTH_LANDMARKS = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
        
        # Tracking variables
        self.ear_history = deque(maxlen=30)  # 1 second at 30 FPS
        self.yawn_history = deque(maxlen=30)
        self.drowsy_frames = 0
        self.alert_active = False
        
    def calculate_ear(self, eye_landmarks):
        """Calculate Eye Aspect Ratio"""
        # Vertical eye landmarks
        A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
        B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
        
        # Horizontal eye landmark
        C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
        
        # EAR calculation
        ear = (A + B) / (2.0 * C)
        return ear
    
    def calculate_mouth_aspect_ratio(self, mouth_landmarks):
        """Calculate Mouth Aspect Ratio for yawn detection"""
        # Vertical distances
        A = dist.euclidean(mouth_landmarks[2], mouth_landmarks[6])
        B = dist.euclidean(mouth_landmarks[3], mouth_landmarks[7])
        C = dist.euclidean(mouth_landmarks[4], mouth_landmarks[8])
        
        # Horizontal distance
        D = dist.euclidean(mouth_landmarks[0], mouth_landmarks[1])
        
        # MAR calculation
        mar = (A + B + C) / (3.0 * D)
        return mar
    
    def extract_eye_region(self, frame, landmarks, eye_indices):
        """Extract eye region for CNN classification"""
        h, w = frame.shape[:2]
        
        # Get eye landmarks
        eye_points = []
        for idx in eye_indices:
            x = int(landmarks.landmark[idx].x * w)
            y = int(landmarks.landmark[idx].y * h)
            eye_points.append([x, y])
        
        eye_points = np.array(eye_points)
        
        # Create bounding box
        x_min, y_min = np.min(eye_points, axis=0)
        x_max, y_max = np.max(eye_points, axis=0)
        
        # Add padding
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        
        # Extract and resize eye region
        eye_region = frame[y_min:y_max, x_min:x_max]
        if eye_region.size > 0:
            eye_region = cv2.resize(eye_region, (128, 128))
            return eye_region / 255.0
        
        return None
    
    def predict_eye_state(self, eye_region):
        """Predict if eye is open or closed using CNN"""
        if eye_region is None:
            return 0.5  # Neutral probability
        
        eye_region = np.expand_dims(eye_region, axis=0)
        prediction = self.model.predict(eye_region, verbose=0)
        return prediction[0][1]  # Probability of closed eye
    
    def detect_drowsiness(self, frame):
        """Main drowsiness detection function"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        drowsiness_status = "AWAKE"
        confidence = 0.0
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            
            # Extract eye regions
            left_eye_region = self.extract_eye_region(frame, landmarks, self.LEFT_EYE_LANDMARKS)
            right_eye_region = self.extract_eye_region(frame, landmarks, self.RIGHT_EYE_LANDMARKS)
            
            # CNN-based eye state prediction
            left_closed_prob = self.predict_eye_state(left_eye_region)
            right_closed_prob = self.predict_eye_state(right_eye_region)
            avg_closed_prob = (left_closed_prob + right_closed_prob) / 2
            
            # EAR calculation as backup
            left_eye_landmarks = [(landmarks.landmark[i].x, landmarks.landmark[i].y) 
                                for i in self.LEFT_EYE_LANDMARKS[:6]]
            right_eye_landmarks = [(landmarks.landmark[i].x, landmarks.landmark[i].y) 
                                 for i in self.RIGHT_EYE_LANDMARKS[:6]]
            
            left_ear = self.calculate_ear(left_eye_landmarks)
            right_ear = self.calculate_ear(right_eye_landmarks)
            avg_ear = (left_ear + right_ear) / 2
            
            # Mouth aspect ratio for yawn detection
            mouth_landmarks = [(landmarks.landmark[i].x, landmarks.landmark[i].y) 
                             for i in self.MOUTH_LANDMARKS[:9]]
            mar = self.calculate_mouth_aspect_ratio(mouth_landmarks)
            
            # Update history
            self.ear_history.append(avg_ear)
            self.yawn_history.append(mar)
            
            # Drowsiness logic
            is_drowsy = False
            
            # Check for closed eyes (CNN + EAR)
            if avg_closed_prob > 0.7 or avg_ear < self.ear_threshold:
                self.drowsy_frames += 1
                if self.drowsy_frames >= 20:  # ~0.67 seconds at 30 FPS
                    is_drowsy = True
                    drowsiness_status = "DROWSY"
                    confidence = max(avg_closed_prob, 1 - avg_ear)
            else:
                self.drowsy_frames = max(0, self.drowsy_frames - 2)
            
            # Check for yawning
            if mar > self.yawn_threshold:
                yawn_count = sum(1 for m in self.yawn_history if m > self.yawn_threshold)
                if yawn_count >= 10:  # Sustained yawning
                    is_drowsy = True
                    drowsiness_status = "YAWNING"
                    confidence = mar
            
            # Extreme drowsiness
            if self.drowsy_frames >= 60:  # 2 seconds
                drowsiness_status = "SLEEPING"
                confidence = 1.0
            
            return {
                'status': drowsiness_status,
                'confidence': confidence,
                'ear': avg_ear,
                'mar': mar,
                'closed_prob': avg_closed_prob,
                'alert_required': is_drowsy
            }
        
        return {
            'status': 'NO_FACE',
            'confidence': 0.0,
            'ear': 0.0,
            'mar': 0.0,
            'closed_prob': 0.0,
            'alert_required': False
        }
