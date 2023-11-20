import cv2
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.utils import normalize
from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
import matplotlib.pyplot as plt

# Setting up the image directory and loading the dataset
image_directory = 'datasets/'

no_tumor_images = os.listdir(image_directory + 'no/')
yes_tumor_images = os.listdir(image_directory + 'yes/')
dataset = []
label = []

INPUT_SIZE = 64  # Adjusted input size for ResNet50

# Looping through the 'no' tumor images
for i, image_name in enumerate(no_tumor_images):
    if image_name.split('.')[1] == 'jpg':
        image = cv2.imread(image_directory + 'no/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(0)

# Looping through the 'yes' tumor images
for i, image_name in enumerate(yes_tumor_images):
    if image_name.split('.')[1] == 'jpg':
        image = cv2.imread(image_directory + 'yes/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(1)

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

dataset = np.array(dataset)
label = np.array(label)

x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=42)

x_train = normalize(x_train, axis=1)
x_test = normalize(x_test, axis=1)

y_train = keras.utils.to_categorical(y_train, num_classes=2)
y_test = keras.utils.to_categorical(y_test, num_classes=2)

# Load pre-trained ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(INPUT_SIZE, INPUT_SIZE, 3))

# Add custom dense layers on top of ResNet50
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=x)

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

model.save('BrainTumorResNet50.h5')

# Print final training and testing accuracies
train_accuracy = history.history['accuracy'][-1]
test_accuracy = history.history['val_accuracy'][-1]
print(f'Final Training Accuracy of ResNet50: {train_accuracy:.4f}')
print(f'Final Testing Accuracy of ResNet50: {test_accuracy:.4f}')

#Model Summary
model.summary()

# Plotting the training and validation accuracy
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.xlabel('ResNet50 Epochs')
plt.ylabel('ResNet50 Accuracy')
plt.legend()
plt.show()
