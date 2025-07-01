import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(description='Convert YawDD dataset to 2 classes')
    parser.add_argument('--input_npz', type=str, required=True, help='Input YawDD .npz file')
    parser.add_argument('--output_npz', type=str, required=True, help='Output .npz file')
    args = parser.parse_args()
    
    data = np.load(args.input_npz)
    X, y = data['X'], data['y']
    
    # Convert 4-class labels to 2 classes:
    # Original: 0=Open, 1=Close, 2=Yawn, 3=No_Yawn
    # New: 0=Open/No_Yawn (alert), 1=Close/Yawn (drowsy)
    indices = np.argmax(y, axis=1)  # Convert one-hot to indices
    new_indices = np.where((indices == 0) | (indices == 3), 0, 1)  # Map classes
    
    # Convert back to one-hot for 2 classes
    new_y = np.zeros((new_indices.size, 2))
    new_y[np.arange(new_indices.size), new_indices] = 1
    
    np.savez_compressed(args.output_npz, X=X, y=new_y)
    print(f"Converted {args.input_npz} to 2 classes. Saved to {args.output_npz}")

if __name__ == '__main__':
    main()
