import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

data = []
labels = []
classes = 43
cur_path = os.getcwd()

#Images and their labels are retrieved in this block. 
for i in range(classes):
    path = os.path.join(cur_path,'train',str(i))
    images = os.listdir(path)

    for a in images:
        try:
            image = Image.open(path + '\\'+ a)
            image = image.resize((30,30))
            image = np.array(image)
            #sim = Image.fromarray(image)
            data.append(image)
            labels.append(i)
        except:
            print("Error in loading image")

# Lists conversion into numpy arrays
data = np.array(data)
labels = np.array(labels)

print(data.shape, labels.shape)

#Splitting training and testing dataset
Y_train, Y_test, x_train, x_test = train_test_split(data, labels, test_size=0.2, random_state=42)

print(Y_train.shape, Y_test.shape, x_train.shape, x_test.shape)

#Converting the labels into one hot encoding
x_train = to_categorical(x_train, 43)
x_test = to_categorical(x_test, 43)

#In this block we will be building the model
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation='softmax'))

#Model compilation
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 15
history = model.fit(Y_train, x_train, batch_size=32, epochs=epochs, validation_data=(Y_test, x_test))
model.save("my_model.h5")

#To easily understand the acccuracy we will plot the graphs. 
plt.figure(0)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.ylabel('epochs')
plt.xlabel('accuracy')
plt.legend()
plt.show()

plt.figure(1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.ylabel('epochs')
plt.xlabel('loss')
plt.legend()
plt.show()

#Here we will check the accuracy on the test dataset that is available
from sklearn.metrics import accuracy_score

x_test = pd.read_csv('Test.csv')

labels = x_test["ClassId"].values
imgs = x_test["Path"].values

data=[]

for img in imgs:
    image = Image.open(img)
    image = image.resize((30,30))
    data.append(np.array(image))

Y_test=np.array(data)

pred = model.predict_classes(X_test)

#Getting accuracy from test dataset.
from sklearn.metrics import accuracy_score
print(accuracy_score(labels, pred))
