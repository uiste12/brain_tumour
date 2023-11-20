import cv2
import os
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Flatten
from keras.models import Model
import matplotlib.pyplot as plt

# Setting up the image directory and loading the dataset
image_directory = 'datasets/'

# Getting the list of images in 'no' and 'yes' folders
no_tumor_images = os.listdir(image_directory + 'no/')
yes_tumor_images = os.listdir(image_directory + 'yes/')

# Lists to store images and labels
dataset = []
label = []

# Adjusted input size for VGG16
INPUT_SIZE = 64

# Loop through 'no' tumor images
for i, image_name in enumerate(no_tumor_images):
    if image_name.split('.')[1] == 'jpg':
        image = cv2.imread(image_directory + 'no/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(0)

# Loop through 'yes' tumor images
for i, image_name in enumerate(yes_tumor_images):
    if image_name.split('.')[1] == 'jpg':
        image = cv2.imread(image_directory + 'yes/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(1)

# Print dataset and label lengths
print('dataset len is {}'.format(len(dataset)))
print('label len is {}'.format(len(label)))

# Visualizing Data
def plot_images(images, labels, num_images=5):
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
    for i in range(num_images):
        axes[i].imshow(images[i])
        axes[i].set_title('')
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

# Visualize some images
plot_images(dataset, label)

# Convert dataset and label to numpy arrays
dataset = np.array(dataset)
label = np.array(label)

# Split dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=42)

# Normalize the input images
x_train = normalize(x_train, axis=1)
x_test = normalize(x_test, axis=1)

# Convert labels to categorical format
y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=2)

# Load pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(INPUT_SIZE, INPUT_SIZE, 3))

# Add custom dense layers on top of VGG16
x = Flatten()(base_model.output)
x = Dense(64, activation='relu')(x)
x = Dense(2, activation='softmax')(x)

model = Model(base_model.input, x)

# Freeze the layers of the pre-trained model
for layer in base_model.layers:
    layer.trainable = False

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(x_train,
                    y_train,
                    batch_size=16,
                    verbose=1,
                    epochs=10,
                    validation_data=(x_test, y_test),
                    shuffle=False)

model.save('BrainTumorVGG16.h5')

# Print final training and testing accuracies
train_accuracy = history.history['accuracy'][-1]
test_accuracy = history.history['val_accuracy'][-1]
print(f'Final Training Accuracy of VGG16: {train_accuracy:.4f}')
print(f'Final Testing Accuracy of VGG16: {test_accuracy:.4f}')

#Model Summary
print('Model Summary of VGG16')
model.summary()

# After the training loop
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.xlabel('VGG16 Epochs')
plt.ylabel('VGG16 Accuracy')
plt.legend()
plt.show()
