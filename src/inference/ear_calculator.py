import numpy as np
from scipy.spatial import distance as dist

class EARCalculator:
    def __init__(self):
        pass
    
    def calculate_ear(self, eye_landmarks):
        if eye_landmarks is None or len(eye_landmarks) < 6:
            return 0.0
        
        A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
        B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
        C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
        
        if C == 0:
            return 0.0
        
        ear = (A + B) / (2.0 * C)
        return ear
    
    def calculate_mar(self, mouth_landmarks):
        if mouth_landmarks is None or len(mouth_landmarks) < 9:
            return 0.0
        
        A = dist.euclidean(mouth_landmarks[2], mouth_landmarks[6])
        B = dist.euclidean(mouth_landmarks[3], mouth_landmarks[7])
        C = dist.euclidean(mouth_landmarks[4], mouth_landmarks[8])
        D = dist.euclidean(mouth_landmarks[0], mouth_landmarks[1])
        
        if D == 0:
            return 0.0
        
        mar = (A + B + C) / (3.0 * D)
        return mar
    
    def smooth_ear(self, ear_values, window_size=5):
        if len(ear_values) < window_size:
            return np.mean(ear_values) if ear_values else 0.0
        
        return np.mean(ear_values[-window_size:])
    
    def detect_blink(self, ear_values, threshold=0.25, consecutive_frames=3):
        if len(ear_values) < consecutive_frames:
            return False
        
        recent_values = ear_values[-consecutive_frames:]
        return all(ear < threshold for ear in recent_values)