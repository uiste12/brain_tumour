# Importing necessary libraries
import cv2
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.optimizers import Adam

# Setting up the image directory and loading the dataset
image_directory='datasets/'
no_tumor_images = os.listdir(image_directory+'no/')
yes_tumor_images = os.listdir(image_directory+'yes/')
dataset= []
label=[]

# Defining the input size for the images
INPUT_SIZE = 64

# Looping through the 'no' tumor images
for i,image_name in enumerate(no_tumor_images):
    if(image_name.split('.')[1] == 'jpg'):
        image = cv2.imread(image_directory+'no/'+image_name)
        image = Image.fromarray(image,'RGB')
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(0)

# Looping through the 'yes' tumor images
for i,image_name in enumerate(yes_tumor_images):
    if(image_name.split('.')[1] == 'jpg'):
        image = cv2.imread(image_directory+'yes/'+image_name)
        image = Image.fromarray(image,'RGB')
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(1)

# Printing dataset and label lengths
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

# Converting dataset and label to numpy arrays
dataset = np.array(dataset)
label = np.array(label)

# Splitting the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=42)

# Normalizing the input images
x_train = normalize(x_train, axis=1)
x_test = normalize(x_test, axis=1)

# Converting labels to categorical format
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

# CNN Model Building
model = Sequential()

model.add(Conv2D(32, (3,3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))

# Compiling the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training the model
history = model.fit(x_train,
                    y_train,
                    batch_size=16,
                    verbose=1,
                    epochs=10,
                    validation_data=(x_test, y_test),
                    shuffle=False)

# Saving the model
model.save('BrainTumor10EpochsCategorical.h5')

# Print final training and testing accuracies
train_accuracy = history.history['accuracy'][-1]
test_accuracy = history.history['val_accuracy'][-1]
print(f'Final Training Accuracy of CNN: {train_accuracy:.4f}')
print(f'Final Testing Accuracy of CNN: {test_accuracy:.4f}')

#Model Summary
print('Model Summary of CNN')
model.summary()

# Plotting the training and validation accuracy
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.xlabel('CNN Epochs')
plt.ylabel('CNN Accuracy')
plt.legend()
plt.show()
