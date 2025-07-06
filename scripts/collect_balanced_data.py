# scripts/collect_balanced_data.py
import cv2
import os
import time
from datetime import datetime

class BalancedDataCollector:
    def __init__(self, output_dir="data/balanced_drowsiness"):
        self.output_dir = output_dir
        os.makedirs(f"{output_dir}/awake", exist_ok=True)
        os.makedirs(f"{output_dir}/drowsy", exist_ok=True)
        
    def collect_awake_samples(self, duration_minutes=10):
        """Collect awake samples - subject should be alert"""
        cap = cv2.VideoCapture(0)
        start_time = time.time()
        count = 0
        
        print("Collecting AWAKE samples - stay alert and look at camera")
        
        while (time.time() - start_time) < duration_minutes * 60:
            ret, frame = cap.read()
            if ret:
                # Save every 30th frame (1 sample per second at 30fps)
                if count % 30 == 0:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    cv2.imwrite(f"{self.output_dir}/awake/awake_{timestamp}.jpg", frame)
                    print(f"Saved awake sample {count//30}")
                count += 1
                
                cv2.imshow('Collecting Awake Data', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def collect_drowsy_samples(self, duration_minutes=10):
        """Collect drowsy samples - subject should simulate drowsiness"""
        cap = cv2.VideoCapture(0)
        start_time = time.time()
        count = 0
        
        print("Collecting DROWSY samples - simulate tiredness, slow blinks, yawning")
        
        while (time.time() - start_time) < duration_minutes * 60:
            ret, frame = cap.read()
            if ret:
                if count % 30 == 0:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    cv2.imwrite(f"{self.output_dir}/drowsy/drowsy_{timestamp}.jpg", frame)
                    print(f"Saved drowsy sample {count//30}")
                count += 1
                
                cv2.imshow('Collecting Drowsy Data', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cap.release()
        cv2.destroyAllWindows()

# Usage
collector = BalancedDataCollector()
collector.collect_awake_samples(10)  # 10 minutes of awake data
collector.collect_drowsy_samples(10)  # 10 minutes of drowsy data
