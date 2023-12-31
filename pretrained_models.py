from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from sklearn.metrics import confusion_matrix, classification_report
from keras.models import load_model
import itertools
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

IMAGE_SIZE = [224, 224]

train_path = '/content/gdrive/MyDrive/Colab Notebooks/train_path'
valid_path = '/content/gdrive/MyDrive/Colab Notebooks/val_path'
test_path = '/content/gdrive/MyDrive/Colab Notebooks/test_path'

train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.xception.preprocess_input) \
    .flow_from_directory(directory=train_path, target_size=(224,224), classes=['class1', 'class2'], batch_size=10)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.xception.preprocess_input) \
    .flow_from_directory(directory=valid_path, target_size=(224,224), classes=['class1', 'class2'], batch_size=10)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.xception.preprocess_input) \
    .flow_from_directory(directory=test_path, target_size=(224,224), classes=['class1', 'class2'], batch_size=10, shuffle=False)

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

train_imgs, train_labels = next(train_batches)

plotImages(train_imgs)
print(train_labels)

xception_1 = Xception(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

for layer in xception_1.layers:
  layer.trainable = False

folders = glob('/content/gdrive/MyDrive/Colab Notebooks/train_path/*')

x = Flatten()(xception_1.output)

prediction = Dense(len(folders), activation='softmax')(x)

model = Model(inputs=xception_1.input, outputs=prediction)

model.summary()

model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(train_path,
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory(valid_path,
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')

r = model.fit(
  training_set,
  validation_data=test_set,
  epochs=10,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)

plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')

model.save('xception_model.h5')

test_imgs, test_labels = next(test_batches)

plotImages(test_imgs)
print(test_labels)

predictions = model.predict(x=test_batches, steps=len(test_batches), verbose=0)

np.round(predictions)

cm = confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(predictions, axis=-1))

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
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
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")
    plt.savefig('Confusion_matrix')

test_batches.class_indices

cm_plot_labels = ['class1', 'class2']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

y_true=test_batches.classes
y_pred=np.argmax(predictions, axis=-1)
target_names = ['class1', 'class2']
print(classification_report(y_true, y_pred, target_names=target_names))
