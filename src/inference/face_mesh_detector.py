import mediapipe as mp
import cv2
import numpy as np

class FaceMeshDetector:
    def __init__(self, max_num_faces=1, refine_landmarks=True, 
                 min_detection_confidence=0.7, min_tracking_confidence=0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        self.LEFT_EYE_LANDMARKS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE_LANDMARKS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.MOUTH_LANDMARKS = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
        
        self.LEFT_EYE_CONTOUR = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_CONTOUR = [362, 385, 387, 263, 373, 380]
    
    def detect_landmarks(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            return results.multi_face_landmarks[0]
        return None
    
    def get_eye_landmarks(self, landmarks, eye_indices):
        if landmarks is None:
            return None
        
        eye_points = []
        for idx in eye_indices:
            x = landmarks.landmark[idx].x
            y = landmarks.landmark[idx].y
            eye_points.append([x, y])
        
        return np.array(eye_points)
    
    def get_mouth_landmarks(self, landmarks):
        if landmarks is None:
            return None
        
        mouth_points = []
        for idx in self.MOUTH_LANDMARKS:
            x = landmarks.landmark[idx].x
            y = landmarks.landmark[idx].y
            mouth_points.append([x, y])
        
        return np.array(mouth_points)
    
    def draw_landmarks(self, frame, landmarks):
        if landmarks is None:
            return frame
        
        h, w = frame.shape[:2]
        
        for idx in self.LEFT_EYE_CONTOUR + self.RIGHT_EYE_CONTOUR:
            x = int(landmarks.landmark[idx].x * w)
            y = int(landmarks.landmark[idx].y * h)
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        
        for idx in self.MOUTH_LANDMARKS:
            x = int(landmarks.landmark[idx].x * w)
            y = int(landmarks.landmark[idx].y * h)
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
        
        return frame