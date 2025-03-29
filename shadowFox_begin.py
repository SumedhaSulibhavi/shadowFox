import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2  # For resizing images
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize the images to a range between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Define the class labels for CIFAR-10 dataset
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Focus on 'cat', 'dog', and 'automobile'
focus_classes = ['cat', 'dog', 'automobile']
focus_class_indices = [class_names.index(c) for c in focus_classes]

# Filter training and test data for only these classes
train_mask = np.isin(train_labels, focus_class_indices)
test_mask = np.isin(test_labels, focus_class_indices)

train_images, train_labels = train_images[train_mask.squeeze()], train_labels[train_mask.squeeze()]
test_images, test_labels = test_images[test_mask.squeeze()], test_labels[test_mask.squeeze()]

# Map labels to their new indices (0 for cat, 1 for dog, 2 for car)
train_labels = np.array([focus_class_indices.index(label[0]) for label in train_labels])
test_labels = np.array([focus_class_indices.index(label[0]) for label in test_labels])

# Initialize the ImageDataGenerator for data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,  # Rotate images randomly by up to 20 degrees
    width_shift_range=0.2,  # Shift images horizontally by up to 20%
    height_shift_range=0.2,  # Shift images vertically by up to 20%
    shear_range=0.2,  # Shear images
    zoom_range=0.2,  # Zoom images in or out
    horizontal_flip=True,  # Flip images horizontally
    fill_mode='nearest'  # Fill missing pixels after transformations
)

# Fit the generator to the training data
datagen.fit(train_images)

# Create the CNN model
model = models.Sequential([
    layers.Input(shape=(32, 32, 3)),  # Input layer with shape (32, 32, 3)
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),  # Dropout layer to prevent overfitting
    layers.Dense(len(focus_classes), activation='softmax')  # Output layer for 3 classes
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model using augmented data
history = model.fit(datagen.flow(train_images, train_labels, batch_size=64),
                    epochs=20, validation_data=(test_images, test_labels))

# Function to display images with better clarity
def show_image(img, title="Image"):
    # Resize the image for better clarity (for display purposes only)
    img_resized = cv2.resize(img, (128, 128))  # Resize to a larger size (128x128)
    plt.figure(figsize=(4, 4))
    plt.imshow(img_resized)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Test the model with the first image in the test set
predictions = model.predict(test_images[:1])
predicted_class = np.argmax(predictions[0])

# Display the first image and its predicted class with better clarity
show_image(test_images[0], title=f"Predicted: {focus_classes[predicted_class]}")

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'Test accuracy: {test_acc}')