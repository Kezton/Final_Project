#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install scikit-learn')
get_ipython().system('pip install --upgrade scikit-learn')
from sklearn.metrics import ConfusionMatrixDisplay


# In[2]:


import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import utils
from sklearn.metrics import confusion_matrix
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[3]:


get_ipython().system('pip install astroNN')


# In[4]:


from astroNN.datasets import load_galaxy10sdss


# In[5]:


images, labels = load_galaxy10sdss()
label = utils.to_categorical(labels, num_classes=10)


# In[6]:


label = label.astype(np.float32)
images = images.astype(np.float32)


# In[7]:


print (labels)
print (label.shape[0])


# In[8]:


train_x,test_x=train_test_split(np.arange(labels.shape[0]),test_size=0.1)
train_images,train_labels,test_images,test_labels=images[train_x],label[train_x],images[test_x],label[test_x]


# In[9]:


print (len(train_x))
print (len(test_x))
print (len(train_labels))


# In[10]:


imageLabel = ["Disk, Face-on, No Spiral", "Smooth, Completely round", "Smooth, in-between round", "Smooth, Cigar shaped",
              "Disk, Edge-on, Rounded Bulge", "Disk, Edge-on, Boxy Bulge", "Disk, Edge-on, No Bulge", "Disk, Face-on, Tight Spiral", "Disk, Face-on, Medium Spiral",
              "Disk, Face-on, Loose Spiral"]


# In[11]:


fig, axes = plt.subplots(ncols = 10, nrows = 10, figsize = (35,30))
index = 0
for i in range(10):
  for j in range(10):
    axes[i,j].set_title(imageLabel[labels[index]])
    axes[i,j].imshow(images[index].astype(np.uint8))
    axes[i,j].get_xaxis().set_visible(False)
    axes[i,j].get_yaxis().set_visible(False)
    index +=1
plt.show()


# In[12]:


plt.imshow(train_images[0].astype(np.uint8))
print (labels[0])
print (label[0])
print (train_images.shape)


# In[13]:


X_train = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in train_images])
X_test = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in test_images])

# Value normalization
X_train  = X_train/255
X_test  = X_test/255


# In[14]:


plt.imshow(X_train[0])


# In[15]:


print(np.shape(X_train))

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

input_shape = (X_train.shape[1], X_train.shape[2], 1)

print(input_shape)


# In[16]:


print (X_train.shape)
print (train_labels.shape)
print (train_labels)


# In[17]:


datagen = ImageDataGenerator(rescale=1./255,
                             rotation_range=90,
                             zoom_range=0.2,
                             horizontal_flip=True,)

datagen.fit(X_train)

datagen.fit(X_test)


# In[25]:


model = Sequential()
model.add(Conv2D(16, (3, 3), activation='tanh', strides=(1, 1),
    padding='same', input_shape=input_shape))
model.add(Conv2D(32, (3, 3), activation='tanh', strides=(1, 1),
    padding='same'))
model.add(MaxPool2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='tanh', strides=(1, 1),
    padding='same'))
model.add(MaxPool2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='tanh', strides=(1, 1),
    padding='same'))
model.add(MaxPool2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='tanh', strides=(1, 1),
    padding='same'))
model.add(MaxPool2D((2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(32, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(len(imageLabel), activation='softmax'))
model.summary()


# In[19]:


model.compile(loss='categorical_crossentropy',
     optimizer='adam',
     metrics=['acc'])

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
batch_size=64
history = model.fit(X_train, train_labels,
                    epochs=30,
                    steps_per_epoch = int(np.ceil(X_train.shape[0]/ float(64))) , batch_size=32, validation_data=(X_test, test_labels), callbacks=[es])


# In[20]:


get_ipython().run_line_magic('matplotlib', 'inline')
fig = plt.gcf()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
fig.savefig('Model_Accuracy.png')


# In[21]:


fig = plt.gcf()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
fig.savefig('Model_Loss.png')


# In[22]:


import itertools
def plot_confusionM(cm, class_names):
    figure = plt.figure(figsize=(15, 15))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

pred = model.predict(X_test)

pred_label = np.argmax(pred, axis=1)
actual_label = np.argmax(test_labels, axis=1)

cm = confusion_matrix(pred_label+1, actual_label+1)
print (cm)
plot_confusionM(cm, imageLabel)


# In[23]:


fig, axes = plt.subplots(ncols=6, nrows=4, sharex=False,
    sharey=True, figsize=(27, 24))
index = 0
for i in range(4):
    for j in range(6):
        axes[i,j].set_title('Actual:' + imageLabel[actual_label[index]] + '\n'
                            + 'Predicted:' + imageLabel[pred_label[index]])
        axes[i,j].imshow(test_images[index].astype(np.uint8), cmap='gray')
        axes[i,j].get_xaxis().set_visible(False)
        axes[i,j].get_yaxis().set_visible(False)
        index += 1
plt.show()


# In[ ]:




