import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
import random
from sklearn.model_selection import train_test_split
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt

# DATA
# Load dataset
dataset = pd.read_csv('driving_log.csv', header=None)
dataset.columns = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
X = dataset['center']
y = dataset['steering']

# DATA PREPROCESSING
def preprocess_data(X, y):
    data_list = []
    label_list = []
    for i in range(len(X)):
        img = cv2.imread(X[i])
        if img is None:
            print(f"Warning: Failed to load image {X[i]}")
            continue
        img = img[60:135, :, :]  # Crop road area
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)  # Convert to YUV
        img = cv2.GaussianBlur(img, (3, 3), 0)  # Apply Gaussian Blur
        img = cv2.resize(img, (200, 66))  # Resize
        img = img / 255
        data_list.append(img)
        label_list.append(y[i])
    data_list = np.array(data_list)
    label_list = np.array(label_list)

    X_train, X_test, y_train, y_test = train_test_split(
        data_list, label_list, test_size=0.3, random_state=42
    )
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = preprocess_data(X, y)

# DATA AUGMENTATION
def augment_image(img, steering_angle):
    h, w = img.shape[:2]

    # Convert from [0, 1] to [0, 255] for OpenCV operations
    img = np.array(img * 255, dtype=np.uint8)

    # Random horizontal flip
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
        steering_angle = -steering_angle

    # Random brightness adjustment
    brightness_scale = 0.4 + np.random.uniform() * 1.2  # Range: [0.4, 1.6]
    img[:, :, 0] = np.clip(img[:, :, 0] * brightness_scale, 0, 255)

    # Random zoom
    zoom_factor = 1 + (random.uniform(-0.2, 0.2))  # Zoom in or out up to ±20%
    new_w = int(w * zoom_factor)
    new_h = int(h * zoom_factor)
    img_zoomed = cv2.resize(img, (new_w, new_h))

    # Crop or pad back to original size
    if zoom_factor < 1:
        pad_w = (w - new_w) // 2
        pad_h = (h - new_h) // 2
        img = cv2.copyMakeBorder(img_zoomed, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_REFLECT_101)
        img = img[:h, :w]
    else:
        crop_x = (new_w - w) // 2
        crop_y = (new_h - h) // 2
        img = img_zoomed[crop_y:crop_y+h, crop_x:crop_x+w]

    # Random pan (translation)
    max_tx = 0.2 * w
    max_ty = 0.2 * h
    tx = random.uniform(-max_tx, max_tx)
    ty = random.uniform(-max_ty, max_ty)
    trans_mat = np.float32([[1, 0, tx], [0, 1, ty]])
    img = cv2.warpAffine(img, trans_mat, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    # 5. Random rotation
    angle = random.uniform(-15, 15)  # ±15 degrees
    rot_mat = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    img = cv2.warpAffine(img, rot_mat, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    return img, steering_angle


# BATCH DATASET
def batch_data(X, y, batch_size=32, augment=False):
    num_samples = len(X)
    indices = np.arange(num_samples)
    while True:
        np.random.shuffle(indices)
        for start_idx in range(0, num_samples, batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            X_batch = []
            y_batch = []
            for i in batch_indices:
                img = X[i]
                angle = y[i]
                if augment:
                    img, angle = augment_image(img, angle)
                X_batch.append(img)
                y_batch.append(angle)
            yield np.array(X_batch), np.array(y_batch)

# MODEL
model = models.Sequential([
    layers.Input(shape=(66, 200, 3)),

    # Conv layers with stride
    layers.Conv2D(24, (5, 5), strides=(2, 2), activation='relu', padding='valid'),
    layers.Conv2D(36, (5, 5), strides=(2, 2), activation='relu', padding='valid'),
    layers.Conv2D(48, (5, 5), strides=(2, 2), activation='relu', padding='valid'),
    layers.Conv2D(64, (3, 3), activation='relu', padding='valid'),
    layers.Conv2D(64, (3, 3), activation='relu', padding='valid'),

    # Flatten and dense layers
    layers.Flatten(),
    layers.Dense(1164, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(50, activation='relu'),
    layers.Dense(10, activation='relu'),
    layers.Dense(1)
])

# Compile with lower learning rate to stabilize training
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mse', metrics=['mae'])

# Train
history = model.fit(
    batch_data(X_train, y_train, batch_size=32, augment=True),
    steps_per_epoch=len(X_train) // 32,
    validation_data=batch_data(X_test, y_test, batch_size=32, augment=False),
    validation_steps=len(X_test) // 32,
    epochs=10
)

# Plot Graphs
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.title('Training and Validation MAE')
plt.show()

# Save Model
model.save('model.h5')
