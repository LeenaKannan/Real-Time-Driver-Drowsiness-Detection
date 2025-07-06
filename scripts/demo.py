# scripts/demo.py
import cv2
import sys
import os
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from main import DrowsinessDetectionSystem

def run_demo():
    """Run drowsiness detection demo"""
    print("🚗 Driver Drowsiness Detection Demo")
    print("=" * 50)
    
    # Check if demo video exists
    demo_video = "tests/demo_videos/demo.mp4"
    
    if os.path.exists(demo_video):
        print(f"Using demo video: {demo_video}")
        video_source = demo_video
    else:
        print("Using live camera (press 'q' to quit)")
        video_source = 0
    
    # Initialize system
    system = DrowsinessDetectionSystem()
    
    try:
        system.start(video_source)
    except KeyboardInterrupt:
        print("\nDemo stopped by user")
    except Exception as e:
        print(f"Demo error: {e}")
    finally:
        system.cleanup()

if __name__ == "__main__":
    run_demo()
