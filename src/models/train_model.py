import os
import argparse
import numpy as np
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Limit GPU memory to 4GB (adjust as needed)
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]
        )
        print(f"Limited GPU memory to 4GB")
    except RuntimeError as e:
        print(e)
        
def build_cnn_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def main():
    parser = argparse.ArgumentParser(description="Train drowsiness detection model")
    parser.add_argument('--train_npz', type=str, required=True, help='Path to training .npz file')
    parser.add_argument('--val_npz', type=str, required=True, help='Path to validation .npz file')
    parser.add_argument('--test_npz', type=str, required=True, help='Path to test .npz file')
    parser.add_argument('--output_dir', type=str, default='models', help='Directory to save trained model')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading training data...")
    train_data = np.load(args.train_npz)
    X_train, y_train = train_data['X'], train_data['y']
    print(f"Training samples: {X_train.shape[0]}")

    print("Loading validation data...")
    val_data = np.load(args.val_npz)
    X_val, y_val = val_data['X'], val_data['y']
    print(f"Validation samples: {X_val.shape[0]}")

    print("Loading test data...")
    test_data = np.load(args.test_npz)
    X_test, y_test = test_data['X'], test_data['y']
    print(f"Test samples: {X_test.shape[0]}")

    # Data augmentation (optional, can be commented out)
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.1),
    ])

    # Apply augmentation to training data
    X_train_aug = data_augmentation(X_train)
    y_train_aug = y_train

    print("Building model...")
    model = build_cnn_model(input_shape=X_train.shape[1:], num_classes=y_train.shape[1])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(args.output_dir, 'best_model.keras'),
            save_best_only=True,
            monitor='val_accuracy',
            mode='max'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2)
    ]

    print("Starting training...")
    history = model.fit(
        X_train_aug, y_train_aug,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks
    )

    print("Evaluating on test data...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc:.4f}")

    print("Saving final model...")
    model.save(os.path.join(args.output_dir, 'drowsiness_detection_model.keras'))

    print("Training complete.")

if __name__ == "__main__":
    main()
