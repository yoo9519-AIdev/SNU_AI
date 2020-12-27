import tensorflow as tf
from tensorflow.python.client import device_lib
from imutils import paths
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
from tensorflow.keras import Model, Sequential
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.metrics as metrics
import random
from glob import glob
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler
import os

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print(tf.__version__)
print(device_lib.list_local_devices())

plt.rcParams["figure.figsize"] = (10, 10)
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.rc('lines', linewidth=3)
plt.rc('font', size=15)

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

DATA_DIR = '../OSA_PSG/new_image'
df = pd.read_csv(f'../OSA_PSG/df2_merge_toy.csv')

glob_image = glob('../OSA_PSG/new_image/*.jpg')
print(len(glob_image))
print(glob_image[:5])

data_image_paths = {os.path.basename(x): x for x in glob_image}
df['path'] = df['filename'].map(data_image_paths.get)
df['ahi_osa'] = df['ahi_osa'].map(lambda x: x.replace('no', 'normal'))

normal_images = glob_image
normal_data = {'path': normal_images, 'ahi_osa': 'Normal'}
df1 = pd.DataFrame(normal_data)
df = pd.concat([df, df1], ignore_index=True, axis=1)
df = df[[6, 8, 10, 11, 12]]
df = df.dropna(axis=0)
df[8] = df[8].replace('normal', 'normal|mild')
df[8] = df[8].replace('mild', 'normal|mild')
df[8] = df[8].replace('moderate', 'moderate|severe')
df[8] = df[8].replace('severe', 'moderate|severe')

label_counts = df[8].value_counts()
print(label_counts)
fig, ax1 = plt.subplots(1,1,figsize = (12, 8))
ax1.bar(np.arange(len(label_counts))+0.5, label_counts)
ax1.set_xticks(np.arange(len(label_counts))+0.5)
_ = ax1.set_xticklabels(label_counts.index, rotation = 0)

all_labels = ['normal|mild', 'moderate|severe']
all_labels = [x for x in all_labels if len(x)>0]
print('All Labels ({}): {}'.format(len(all_labels), all_labels))
for c_label in all_labels:
    if len(c_label)>1: # leave out empty labels
        df[c_label] = df[8].map(lambda finding: 1 if c_label in finding else 0)
df['osa_vec'] = df.apply(lambda x: [x[all_labels].values], 1).map(lambda x: x[0])
print(df)

train_df = df[df[11] == 'train']
valid_df = df[df[11] == 'valid']
test_df = df[df[11] == 'test']
print(train_df)

BATCH_SIZE = 64

datagen = ImageDataGenerator(rescale = 1./255,
                             zoom_range=0.1,
                             height_shift_range=0.05,
                             horizontal_flip=True,
                             samplewise_std_normalization=True,
                             samplewise_center=True,
                             width_shift_range=0.05,
                             rotation_range=5)

test_datagen = ImageDataGenerator(rescale = 1./255)

train_gen = datagen.flow_from_dataframe(train_df,
                                        directory=None,
                                        x_col=12,
                                        y_col=8,
                                        target_size=(512, 512),
                                        color_mode='grayscale',
                                        batch_size=BATCH_SIZE,
                                        class_mode='binary',
                                        shuffle=True)

valid_gen = test_datagen.flow_from_dataframe(valid_df,
                                           directory=None,
                                           x_col=12,
                                           y_col=8,
                                           target_size=(512, 512),
                                           color_mode='grayscale',
                                           batch_size=BATCH_SIZE,
                                           class_mode='binary',
                                           shuffle=False)

test_gen = test_datagen.flow_from_dataframe(test_df,
                                            directory=None,
                                            x_col=12,
                                            y_col=8,
                                            target_size=(512, 512),
                                            color_size='grayscale',
                                            batch_size=BATCH_SIZE,
                                            class_mode='binary',
                                            shuffle=False)

train_data = tf.data.Dataset.from_generator(lambda: train_gen,
                                            output_types=(tf.float32, tf.int32),
                                            output_shapes=([None, 512, 512, 1], [None, ]))

valid_data = tf.data.Dataset.from_generator(lambda: valid_gen,
                                          output_types=(tf.float32, tf.int32),
                                          output_shapes=([None, 512, 512, 1], [None, ]))

test_data = tf.data.Dataset.from_generator(lambda: test_gen,
                                           output_types=(tf.float32, tf.int32),
                                           output_shapes=([None, 512, 512, 1], [None, ]))


def osa_layer():

    # Model input
    input_layer = layers.Input(shape=(512, 512, 1), name='input')

    # First block
    x = layers.Conv2D(filters=64, kernel_size=3,
                      activation='relu', padding='same',
                      name='conv2d_1')(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(pool_size=2, name='maxpool2d_1')(x)
    # x = layers.Dropout(0.1, name='dropout_1')(x)

    # Second block
    x = layers.Conv2D(filters=96, kernel_size=3,
                      activation='relu', padding='same',
                      name='conv2d_2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(pool_size=2, name='maxpool2d_2')(x)
    # x = layers.Dropout(0.1, name='dropout_2')(x)

    # Third block
    x = layers.Conv2D(filters=128, kernel_size=3,
                      activation='relu', padding='same',
                      name='conv2d_3')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(pool_size=2, name='maxpool2d_3')(x)
    # x = layers.Dropout(0.1, name='dropout_3')(x)

    # Fourth block
    x = layers.Conv2D(filters=160, kernel_size=3,
                      activation='relu', padding='same',
                      name='conv2d_4')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(pool_size=2, name='maxpool2d_4')(x)
    # x = layers.Dropout(0.1, name='dropout_4')(x)

    # Fifth block
    x = layers.Conv2D(filters=192, kernel_size=3,
                      activation='relu', padding='same',
                      name='conv2d_5')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(pool_size=2, name='maxpool2d_5')(x)
    # x = layers.Dropout(0.1, name='dropout_5')(x)

    # Sixth block
    x = layers.Conv2D(filters=224, kernel_size=3,
                      activation='relu', padding='same',
                      name='conv2d_6')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(pool_size=2, name='maxpool2d_6')(x)
    # x = layers.Dropout(0.1, name='dropout_6')(x)

    # Seventh block
    x = layers.Conv2D(filters=256, kernel_size=3,
                      activation='relu', padding='same',
                      name='conv2d_7')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(pool_size=2, name='maxpool2d_7')(x)
    # x = layers.Dropout(0.1, name='dropout_7')(x)

    # Eighth block
    x = layers.Conv2D(filters=288, kernel_size=3,
                      activation='relu', padding='same',
                      name='conv2d_8')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(pool_size=2, name='maxpool2d_8')(x)
    # x = layers.Dropout(0.1, name='dropout_8')(x)

    # Ninth block
    x = layers.Conv2D(filters=320, kernel_size=3,
                      activation='relu', padding='same',
                      name='conv2d_9')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(pool_size=2, name='maxpool2d_9')(x)
    # x = layers.Dropout(0.1, name='dropout_9')(x)

    # Tenth block
    x = layers.Conv2D(filters=352, kernel_size=3,
                      activation='relu', padding='same',
                      name='conv2d_10')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(pool_size=2, name='maxpool2d_10')(x)

    # Pooling and output
    x = layers.GlobalAveragePooling2D(name='global_average_pooling2d')(x)
    output = layers.Dense(units=1,
                          activation='sigmoid',
                          name='output')(x)

    model = Model(input_layer, output)

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model

model = osa_layer()
model.summary()

def lr_decay(epoch):
    initial_lr = 0.001
    lr = initial_lr * np.exp(-0.1 * epoch)
    return lr

lr_scheduler = LearningRateScheduler(lr_decay, 1)
csv_logger = CSVLogger(filename='9-layer_double_adam_512_aug_bn_dropout01_explr.csv')
model_checkpoint = ModelCheckpoint(filepath='7-layer_double_adam_512_aug_bn_dropout01_explr_{epoch:04d}.hdf5')

train_steps = train_gen.samples // BATCH_SIZE
valid_steps = valid_gen.samples // BATCH_SIZE

history = model.fit(train_data,
                    epochs=50,
                    steps_per_epoch=train_steps,
                    validation_data=(valid_data),
                    validation_steps=valid_steps,
                    shuffle=False,
                    callbacks=[lr_scheduler, csv_logger, model_checkpoint])
