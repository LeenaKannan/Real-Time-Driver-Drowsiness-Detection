import cv2
import time
import numpy as np

class VideoProcessor:
    def __init__(self):
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
    
    def initialize_camera(self, source=0, width=640, height=480, fps=30):
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        return cap
    
    def calculate_fps(self):
        self.fps_counter += 1
        if self.fps_counter % 30 == 0:
            self.current_fps = 30 / (time.time() - self.fps_start_time)
            self.fps_start_time = time.time()
        return self.current_fps
    
    def preprocess_frame(self, frame):
        if frame is None:
            return None
        
        frame = cv2.flip(frame, 1)
        return frame
    
    def draw_fps(self, frame, fps):
        cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)