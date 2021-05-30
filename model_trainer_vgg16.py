import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.engine import sequential
import cv2
import numpy as np

from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from PIL import Image

# Create a sample Convolutional Neural Network(CNN)
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D

# Saving and loading model and weights
from keras.models import model_from_json
from keras.models import load_model

import argparse

# Define the parser
parser = argparse.ArgumentParser(description='Input args')
parser.add_argument('--training_source', action="store", dest='training_source', default='training_data/images/')
parser.add_argument('--img_size', action="store", dest='img_size', default='224')
parser.add_argument('--dataset_size', action="store", dest='dataset_size', default='450')
args = parser.parse_args()

# Prepare training/testing data
# load and prepare data
print('\n\n')
print('====================/Arguments/====================')
training_source =args.training_source
print('Training source: ' + training_source)
ids = []
training_sequence_file = open("models/training_sequence.txt", "w")
for person_id in os.listdir(training_source):
  ids.append(str(person_id))
  training_sequence_file.write(str(person_id) + '\n')
num_classes = len(ids)
print('classes to train: ' + str(num_classes))
training_sequence_file.close()
img_size = int(args.img_size)
print('Training images size: ' + str(img_size))
dataset_size = int(args.dataset_size)
print('Number of images per class: ' + str(dataset_size))
print('====================/Arguments/====================')
print('\n')

img_data_list = []
labels = []
valid_images = [".jpg",".gif",".png"]

def resample_image(img):
  img = Image.fromarray(img.astype('uint8'), 'L')
  img = img.resize((img_size,img_size), Image.ANTIALIAS)
  return np.array(img)

# loop in folder images to get all image.
for index, id in enumerate(ids):
  dir_path = 'training_data/images/' + id
  count=0
  for img_path in os.listdir(dir_path):
    name, ext = os.path.splitext(img_path)
    if ext.lower() not in valid_images:
      continue
    if(count>dataset_size-1):
      break
    # use cv2.imread to read and convert to gray color.
    img_data = cv2.imread(dir_path + '/' + img_path)
    # convert image to gray
    img_data=cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
    img_data = resample_image(img_data)
    img_data = img_data.astype('float32')
    # store images and label of people in array list.
    img_data_list.append(img_data)
    labels.append(index)
    count=count+1

# convert python array to numpy array
# and divde with 255.0 to improve training performance.
img_data = np.array(img_data_list)

labels = np.array(labels ,dtype='int64')
# scale down(so easy to work with)
img_data /= 255.0

#expand dims for conv2d
new_img_data=[]
for image in img_data:
  image=image[:,:,np.newaxis]
  new_img_data.append(image)
new_img_data=np.array(new_img_data)

# shuffle dataset for improving training accuracy.
# convert class labels to one-hot encoding
Y = np_utils.to_categorical(labels, num_classes)

#Shuffle the dataset
x,y = shuffle(new_img_data,Y, random_state=2)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
# end of Prepare training/testing data

# Defining the model
print('====================/Defining the model/====================')
input_shape=new_img_data[0].shape
print('Input_shape: ' + str(input_shape))

# VGG16 architecture
model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', padding="SAME", input_shape = input_shape))
model.add(Conv2D(64, (3, 3), activation='relu', padding="SAME"))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(Conv2D(128, (3, 3), activation='relu', padding="SAME"))
model.add(Conv2D(128, (3, 3), activation='relu', padding="SAME"))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Conv2D(256, (3, 3), activation='relu', padding="SAME"))
model.add(Conv2D(256, (3, 3), activation='relu', padding="SAME"))
model.add(Conv2D(256, (3, 3), activation='relu', padding="SAME"))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Conv2D(512, (3, 3), activation='relu', padding="SAME"))
model.add(Conv2D(512, (3, 3), activation='relu', padding="SAME"))
model.add(Conv2D(512, (3, 3), activation='relu', padding="SAME"))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(Conv2D(512, (3, 3), activation='relu', padding="SAME"))
model.add(Conv2D(512, (3, 3), activation='relu', padding="SAME"))
model.add(Conv2D(512, (3, 3), activation='relu', padding="SAME"))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Flatten())

model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compile model
epochs = 11
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())
print('====================/Defining the model/====================')
print('\n')

print('====================/Traning model/====================')
# train model with dataset and then save it into disk for later use.
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# serialize model to JSON
model_json = model.to_json()
with open("models/model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("models/model.h5")
print("Saved model to disk")
print('====================/Traning model/====================')
print('\n\n')