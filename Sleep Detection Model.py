import argparse
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (256, 256))
    img = img / 255.0  # Normalize
    return np.expand_dims(img, axis=0)

def train_model(data_dir):
    # Data loading and preprocessing
    data = image_dataset_from_directory(data_dir, image_size=(256, 256), batch_size=32)
    data = data.map(lambda x, y: (x / 255.0, y))

    # Split data
    train_size = int(len(data) * 0.7)
    val_size = int(len(data) * 0.2)
    test_size = int(len(data) * 0.1)

    train_data = data.take(train_size)
    val_data = data.skip(train_size).take(val_size)
    test_data = data.skip(train_size + val_size).take(test_size)

    # Model definition
    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=(256, 256, 3)),
        MaxPooling2D(),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(),
        Conv2D(16, (3, 3), activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(256, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    # Training
    hist = model.fit(train_data, epochs=20, validation_data=val_data)

    # Save the model
    model.save(os.path.join('models', 'sleepclassificationmodel.h5'))
    print("Model trained and saved.")

    # Plot training history
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(hist.history['loss'], label='loss', color='teal')
    ax[0].plot(hist.history['val_loss'], label='val_loss', color='orange')
    ax[0].set_title('Loss')
    ax[0].legend()

    ax[1].plot(hist.history['accuracy'], label='accuracy', color='teal')
    ax[1].plot(hist.history['val_accuracy'], label='val_accuracy', color='orange')
    ax[1].set_title('Accuracy')
    ax[1].legend()

    plt.show()

def predict_image(model_path, image_path):
    model = load_model(model_path)
    img = load_and_preprocess_image(image_path)
    prediction = model.predict(img)
    if prediction > 0.5:
        print("Predicted class is Sleeping")
    else:
        print("Predicted class is Not Sleeping")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sleep Detection Model')
    parser.add_argument('--train', type=str, help='Path to the training data directory')
    parser.add_argument('--predict', type=str, help='Path to the image for prediction')
    args = parser.parse_args()

    if args.train:
        train_model(args.train)
    elif args.predict:
        predict_image(os.path.join('models', 'sleepclassificationmodel.h5'), args.predict)
    else:
        print("Please specify --train or --predict argument.")
