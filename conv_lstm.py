from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)

# Commented out IPython magic to ensure Python compatibility.
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from keras import regularizers
from keras.applications.xception import preprocess_input
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# %matplotlib inline

input_shape = (224, 224, 3)

train_path = '/content/gdrive/MyDrive/Colab Notebooks/train_path'
valid_path = '/content/gdrive/MyDrive/Colab Notebooks/val_path'
test_path = '/content/gdrive/MyDrive/Colab Notebooks/test_path'

#just import vgg16, vgg19, or resnet in place of xception 
#train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet.preprocess_input) \
 #   .flow_from_directory(directory=train_path, target_size=(224,224), classes=['class1', 'class2'], batch_size=3045)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.xception.preprocess_input) \
    .flow_from_directory(directory=valid_path, target_size=(224,224), classes=['class1', 'class2'], batch_size=380)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.xception.preprocess_input) \
    .flow_from_directory(directory=test_path, target_size=(224,224), classes=['class1', 'class2'], batch_size=380, shuffle=False)

def augment_data(train_path, batch_size):
    # Define an ImageDataGenerator for data augmentation
    train_datagen = ImageDataGenerator(
        #rescale=1./255,
        #rotation_range=20,
        #width_shift_range=0.2,
        #height_shift_range=0.2,
        #shear_range=0.2,
        #zoom_range=0.2,
        horizontal_flip=True,
        #vertical_flip=True,
        #fill_mode='nearest',
        preprocessing_function=tf.keras.applications.xception.preprocess_input
    )

    # Load the training data with the ImageDataGenerator and return the batch
    train_batches = train_datagen.flow_from_directory(
        directory=train_path,
        target_size=(224,224),
        classes=['class1', 'class2'],
        batch_size=batch_size
    )

    return train_batches


train_batches = augment_data(train_path, 3045)

x_train, y_train = train_batches.next()

x_train = np.expand_dims(x_train, axis=1)
y_train = np.expand_dims(y_train, axis=1)

x_train.shape

y_train.shape

x_val, y_val = valid_batches.next()

x_test, y_test = test_batches.next()

x_val = np.expand_dims(x_val, axis=1)
y_val = np.expand_dims(y_val, axis=1)

x_test = np.expand_dims(x_test, axis=1)
y_test = np.expand_dims(y_test, axis=1)

model = Sequential([
    TimeDistributed(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', padding='same'), input_shape=(None, 224, 224, 3)),
    TimeDistributed(BatchNormalization()),
    TimeDistributed(MaxPool2D(pool_size=(2, 2), strides=2)),
    TimeDistributed(Dropout(0.25)),
    TimeDistributed(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')),
    TimeDistributed(BatchNormalization()),
    TimeDistributed(MaxPool2D(pool_size=(2, 2), strides=2)),
    TimeDistributed(Dropout(0.25)),
    TimeDistributed(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')),
    TimeDistributed(BatchNormalization()),
    TimeDistributed(MaxPool2D(pool_size=(2, 2), strides=2)),
    TimeDistributed(Dropout(0.25)),
    TimeDistributed(Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')),
    TimeDistributed(BatchNormalization()),
    TimeDistributed(MaxPool2D(pool_size=(2, 2), strides=2)),
    TimeDistributed(Dropout(0.25)),
    TimeDistributed(Flatten()),
    LSTM(units=256, activation='relu', return_sequences=True),
    LSTM(units=128, activation='relu', return_sequences=True),
    Dense(units=512, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5,l2=1e-4), bias_regularizer = regularizers.l2(1e-4),
          activity_regularizer = regularizers.l2(1e-5)),
    Dense(units=512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(units=5, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=32, epochs=35, validation_data=(x_val, y_val))

model.save('leaves_lstm_model.h5')

plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')

predictions = model.predict(x=x_test, steps=len(x_test), verbose=0)

np.round(predictions)

from sklearn.metrics import confusion_matrix, classification_report
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
    plt.savefig('Confusion matrix')

test_batches.class_indices

import itertools
cm_plot_labels = ['class1', 'class2']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

y_true=test_batches.classes
y_pred=np.argmax(predictions, axis=-1)
target_names = ['class1', 'class2']
print(classification_report(y_true, y_pred, target_names=target_names))

