import os, shutil
import tensorflow as tf
import keras
keras.__version__
import numpy
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras import layers
from keras import models
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.constraints import MaxNorm
from keras.callbacks import ModelCheckpoint
from keras import utils
from keras.utils import load_img, img_to_array
from keras import optimizers
from keras.preprocessing import image
from keras.src.legacy.preprocessing.image import ImageDataGenerator

 # Prefetched training image augmentation and shuffling.  Drastically increases throughput removing augmentation cpu bottleneck.

sections = {}
categories = {}

base_dir = 'dataset_240' # filepath to the folder the data is in or just the folder name if this python file is in the same folder as the dataset base folder
# This python file will read the image dimensions and set the model to them so you can switch between 'dataset_480' or 'dataset_240' just by switching
# the base_dir variable as long as the model specified will run on the image dimensions.

for x in os.listdir(base_dir):
    sections[x+'_dir'] = os.path.join(base_dir, x)
for y in os.listdir(base_dir):
    for z in os.listdir(sections[y+'_dir']):
        categories[y+'_'+z+'_dir'] = os.path.join(sections[y+'_dir'], z)
        print('total '+y+' '+z+' images: ', len(os.listdir(categories[y+'_'+z+'_dir'])))
total_categories = int(len(categories)/3)
print("categories: "+str(total_categories))
height, width, channels = img_to_array(load_img(os.path.join(next(iter(categories.values())), next(os.scandir(next(iter(categories.values())))).name))).shape
print("Width: "+str(width)+", Height: "+str(height))

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(height, width, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.2))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.2))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.2))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.2))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(512, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.2))
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.BatchNormalization())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.BatchNormalization())
model.add(layers.Dense(total_categories, activation='softmax'))
model.summary()
model.compile(loss='sparse_categorical_crossentropy',
              metrics=['acc'],
              optimizer='adam'
              )

train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    sections['train_dir'],
    target_size=(height, width),
    batch_size=25,
    class_mode='sparse',
    shuffle=False
)

file_paths = train_generator.filepaths
labels = train_generator.labels

dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))

data_augmentation = keras.Sequential([
    layers.Rescaling(1./255),
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.1),
    layers.RandomShear(0.1),
    layers.RandomTranslation(0.1, 0.1),
])

def load_raw_png(file_path, label):
    image_raw = tf.io.read_file(file_path)
    image = tf.io.decode_png(image_raw, channels=3)
    return image, label

AUTOTUNE = tf.data.AUTOTUNE
dataset = dataset.shuffle(buffer_size=10000)
dataset = dataset.map(load_raw_png, num_parallel_calls=AUTOTUNE)
dataset = dataset.batch(25)
dataset = dataset.map(
    lambda x, y: (data_augmentation(x, training=True), y),
    num_parallel_calls=AUTOTUNE
)
dataset = dataset.prefetch(buffer_size=AUTOTUNE)

validation_generator = test_datagen.flow_from_directory(
        sections['validation_dir'],
        target_size=(height, width),
        batch_size=25,
        class_mode='sparse')
for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break
history = model.fit(
      x=dataset,
      epochs=200,
      validation_data=validation_generator)
test_generator = test_datagen.flow_from_directory(
    sections['test_dir'],
    target_size=(height, width),
    batch_size=25,
    class_mode='sparse')
test_loss, test_acc = model.evaluate(test_generator, steps=4*total_categories)
print('test acc:', test_acc)
model.save('fauna_240-25.keras')
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
