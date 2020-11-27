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
from tensorflow.keras.applications import EfficientNetB7
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score, average_precision_score

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
df[11] = df[11].replace("test", "train")
print(df)

train_df = df[df[11] == 'train']
valid_df = df[df[11] == 'valid']
print(train_df)


# ----------------------------------------Define---------------------------------------- #


BATCH_SIZE = 32

datagen = ImageDataGenerator(rescale = 1./255,
                             zoom_range=0.1,
                             height_shift_range=0.05,
                             horizontal_flip=True,
                             samplewise_std_normalization=True,
                             samplewise_center=True,
                             width_shift_range=0.05)

test_datagen = ImageDataGenerator(rescale = 1./255)

train_gen = datagen.flow_from_dataframe(train_df,
                                        directory=None,
                                        x_col=12,
                                        y_col=8,
                                        target_size=(224, 224),
                                        color_mode='grayscale',
                                        batch_size=BATCH_SIZE,
                                        class_mode='binary',
                                        shuffle=True)

test_gen = test_datagen.flow_from_dataframe(valid_df,
                                           directory=None,
                                           x_col=12,
                                           y_col=8,
                                           target_size=(224, 224),
                                           color_mode='grayscale',
                                           batch_size=BATCH_SIZE,
                                           class_mode='binary',
                                           shuffle=False)

test_X, test_Y = next(test_datagen.flow_from_dataframe(valid_df,
                                                  directory=None,
                                                  x_col=12,
                                                  y_col=8,
                                                  target_size=(224, 224),
                                                  color_mode='grayscale',
                                                  batch_size=BATCH_SIZE,
                                                  class_mode='binary',
                                                  suffle=False))

train_data = tf.data.Dataset.from_generator(lambda: train_gen,
                                            output_types=(tf.float32, tf.int32),
                                            output_shapes=([None, 224, 224, 1], [None, ]))

valid_data = tf.data.Dataset.from_generator(lambda: test_gen,
                                          output_types=(tf.float32, tf.int32),
                                          output_shapes=([None, 224, 224, 1], [None, ]))



# ---------------------------------------- Network ---------------------------------------- #


labels = ['normal|mild', 'moderate|severe']
image_size = 224

EffiNet7 = EfficientNetB7(
    include_top=False,
    weights="imagenet",
    input_shape=(image_size, image_size, 3),
    classifier_activation='sigmoid'
)

EffiNet7.trainable = True

model = tf.keras.Sequential([
    EffiNet7,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1, activation='sigmoid')])

model.compile(
    optimizer='adam',
    loss = 'binary_crossentropy',
    metrics=['AUC', 'Precision', 'Recall']
)


# ---------------------------------------- Define_learn_epoch ---------------------------------------- #


LR_START = 0.00001
LR_MAX = 0.00005
LR_MIN = 0.00001
LR_RAMPUP_EPOCHS = 5
LR_SUSTAIN_EPOCHS = 0
LR_EXP_DECAY = .8


def lrfn(epoch):
    if epoch < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY ** (epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN
    return lr


lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)

rng = [i for i in range(25)]
y = [lrfn(x) for x in rng]


def get_callbacks(model_name):
    callbacks = []
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=f'model.{model_name}.h5',
        verbose=1,
        save_best_only=True)

    erly = tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=20,
                                            verbose=1)
    callbacks.append(checkpoint)
    callbacks.append(erly)
    callbacks.append(lr_callback)
    return callbacks


train_steps = train_gen.samples // BATCH_SIZE
test_steps = test_gen.samples // BATCH_SIZE

callbacks = get_callbacks('EfficientNetB7')
history = model.fit(train_gen,
                   steps_per_epoch = train_steps,
                   validation_data = test_gen,
                   validation_steps = test_steps,
                   epochs = 50,
                   callbacks = callbacks)


# ---------------------------------------- Result ---------------------------------------- #


y_pred = model.predict(test_X)

for label, p_count, t_count in zip(labels,
                                     100 * np.mean(y_pred, 0),
                                     100 * np.mean(test_Y, 0)):
    print('%s: actual: %2.2f%%, predicted: %2.2f%%' % (label, t_count, p_count))


fig, c_ax = plt.subplots(1, 1, figsize=(9, 9))

for (idx, c_label) in enumerate(labels):
    fpr, tpr, thresholds = roc_curve(test_Y[:, idx].astype(int), y_pred[:, idx])
    c_ax.plot(fpr, tpr, label='%s (AUC:%0.2f)' % (c_label, auc(fpr, tpr)))

c_ax.legend()
c_ax.set_xlabel('False Positive Rate')
c_ax.set_ylabel('True Positive Rate')
fig.savefig('trained_net.png')

print('ROC auc score: {:.3f}'.format(roc_auc_score(test_Y.astype(int), y_pred)))