## Fashion MNIST

### Import Libraries
```
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import itertools
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from __future__ import absolute_import, division, print_function, unicode_literals
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
```
### Reshape images
```
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))
train_images, test_images = train_images / 255.0, test_images / 255.0
```
```
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
```
```
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
```
```
img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)

cnn = Sequential()
cnn.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Dropout(0.2))

cnn.add(Flatten())

cnn.add(Dense(128, activation='relu'))
cnn.add(Dense(10, activation='softmax'))

cnn.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
```
```
history = cnn.fit(train_images, train_labels,
          batch_size=256,
          epochs=10,
          verbose=1)
          
score = cnn.evaluate(test_images, test_labels, verbose=0)
print('Test loss:', score1[0])
print('Test accuracy:', score1[1])
              
predicted_classes = cnn.predict(test_images)
```
## Plot Confusion Matrix
```
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
```
```
# Predict the values from the validation dataset
ypred_onehot = cnn.predict(test_images)
# Convert predictions classes from one hot vectors to labels: [0 0 1 0 0 ...] --> 2
ypred = np.argmax(ypred_onehot,axis=1)
# Convert validation observations from one hot vectors to labels
ytrue = np.argmax(test_labels,axis=1)
# compute the confusion matrix
confusion_mtx = confusion_matrix(ytrue, ypred)
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes=class_names)
```
## Defining model architecture
```
img_inputs = keras.Input(shape=(28, 28, 1))

x = keras.layers.Conv2D(32, 3, strides=(1,1), activation='relu', input_shape=(28,28,1))(img_inputs)
x = keras.layers.MaxPooling2D(pool_size=(2, 2),strides=2)(x)
x = keras.layers.Flatten()(x)

x2 = keras.layers.Flatten()(img_inputs)
x2 = keras.layers.Dense(64, activation='relu')(x2)

x3 = keras.layers.concatenate([x,x2])
x3 = keras.layers.Dense(32, activation='relu')(x3)
outputs = keras.layers.Dense(10, activation = 'softmax')(x3)
model2 = keras.Model(inputs=img_inputs, outputs=outputs, name='mnist_model')
model2.summary()
```
## Plot model as png
```
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
plot_model(model2, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
SVG(model_to_dot(model2, show_shapes=True, show_layer_names=True).create(prog='dot',format='svg'))
```
