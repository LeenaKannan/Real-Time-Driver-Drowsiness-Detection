import numpy as np

# Load the full training dataset
train_data = np.load('data/processed/mrl_processed.npz')
X_train_full = train_data['X']
y_train_full = train_data['y']

# Reduce training size to 10% (adjust as needed)
reduced_size = int(0.1 * X_train_full.shape[0])  # 10% of the samples
X_train_reduced = X_train_full[:reduced_size]
y_train_reduced = y_train_full[:reduced_size]

# Save the reduced dataset
np.savez_compressed('data/processed/mrl_processed_reduced.npz', X=X_train_reduced, y=y_train_reduced)
