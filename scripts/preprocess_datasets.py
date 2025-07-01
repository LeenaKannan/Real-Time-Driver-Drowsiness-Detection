import os
import numpy as np
import argparse
from src.models.data_preprocessing import DataPreprocessor

def main():
    parser = argparse.ArgumentParser(description='Preprocess datasets for driver drowsiness detection')
    parser.add_argument('--mrl_path', type=str, default='data/raw/MRL_dataset',
                        help='Path to MRL dataset')
    parser.add_argument('--yawdd_path', type=str, default='data/raw/YawDD_dataset/train',
                        help='Path to YawDD dataset (already processed frames)')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                        help='Output directory for processed datasets')
    args = parser.parse_args()

    preprocessor = DataPreprocessor(target_size=(128, 128))
    os.makedirs(args.output_dir, exist_ok=True)

    # Process MRL Dataset
    print("="*50)
    print("Processing MRL Dataset")
    print("="*50)
    X_mrl, y_mrl = preprocessor.load_mrl_dataset(args.mrl_path)
    np.savez_compressed(os.path.join(args.output_dir, 'mrl_processed.npz'), 
                         X=X_mrl, y=y_mrl)
    print(f"Saved MRL data: X shape {X_mrl.shape}, y shape {y_mrl.shape}")

    # Process YawDD Dataset
    print("\n" + "="*50)
    print("Processing YawDD Dataset")
    print("="*50)
    X_yawdd, y_yawdd = preprocessor.load_yawdd_frames(args.yawdd_path)
    np.savez_compressed(os.path.join(args.output_dir, 'yawdd_processed.npz'), 
                         X=X_yawdd, y=y_yawdd)
    print(f"Saved YawDD data: X shape {X_yawdd.shape}, y shape {y_yawdd.shape}")

    print("\n" + "="*50)
    print("Preprocessing Complete")
    print("="*50)

if __name__ == "__main__":
    main()
