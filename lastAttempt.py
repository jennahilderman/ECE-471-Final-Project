import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import cv2
import tensorflow as tf

import keras
from keras import layers, metrics, losses, preprocessing
from keras.preprocessing import sequence
from keras.losses import SparseCategoricalCrossentropy, MeanSquaredError
from keras.models import Sequential
import pathlib
import pandas as pd
from PIL import Image 
from PIL.ImageDraw import Draw
import xml.etree.ElementTree as ET

red_label = 3
black_label = 1
blue_label = 2




TFRECORD_TRAINING = 'training_demo\\models\\train\\duck.tfrecord'
TRAINING_LABEL_MAP = 'training_demo\\models\\train\\duck_label_map.pbtxt'

TFRECORD_VALIDATION = 'training_demo\\models\\test\\duck.tfrecord'
VALIDATION_LABEL_MAP = 'training_demo\\models\\test\\duck_label_map.pbtxt'


num_classes = 3




def parse_record(data_record):
    # Parse the tfRecord into its features
    feature = {'image/encoded': tf.io.FixedLenFeature([], tf.string),
               'image/object/class/label': tf.io.VarLenFeature(tf.int64),
               'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
               'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
               'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
               'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
               'image/filename': tf.io.FixedLenFeature([], tf.string)
               }
    return tf.io.parse_single_example(data_record, feature)


def get_records(path):
    # Get the dataset
    dataset = tf.data.TFRecordDataset([path])
    # Get features
    record_iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    # Get the number if items
    num_records = dataset.reduce(np.int64(0), lambda x, _: x + 1).numpy()
    np_images = []
    labels = []
    targets = []


    for i in range(num_records):
        parsed_example = parse_record(record_iterator.get_next())
        encoded_image = parsed_example['image/encoded']
        np_image = np.asarray(tf.image.decode_image(encoded_image, channels=3))
        label =  np.asarray(tf.sparse.to_dense(parsed_example['image/object/class/label'], default_value=0))
        xmin =  np.asarray(tf.sparse.to_dense( parsed_example['image/object/bbox/xmin'], default_value=0))
        xmax =  np.asarray(tf.sparse.to_dense( parsed_example['image/object/bbox/xmax'], default_value=0))
        ymin =  np.asarray(tf.sparse.to_dense( parsed_example['image/object/bbox/ymin'], default_value=0))
        ymax =  np.asarray(tf.sparse.to_dense( parsed_example['image/object/bbox/ymax'], default_value=0))
        bgr = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
        np_images.append(list(bgr))
        labels.append(list(label))
        targets.append(list((xmin, ymin, xmax, ymax)))
        height, width = np.shape(np_image[:, :, 1])
    return np_images, labels, targets, height, width

train_images, train_labels, train_targets, height, width = get_records(TFRECORD_TRAINING)
# padded = preprocessing.sequence.pad_sequences(train_images)
train_images =np.expand_dims(train_images, axis=-1).tolist()
train_labels =np.expand_dims(train_labels, axis=-1).tolist()
train_targets = np.expand_dims(train_targets, axis=-1).tolist()


print(np.asarray(train_images).shape)
print(np.asanyarray(train_labels).shape)
print(np.asanyarray(train_targets).shape)
print(height, width)

validation_images, validation_labels, validation_targets, height, width = get_records(TFRECORD_VALIDATION)

validation_images = np.expand_dims(validation_images, axis=-1).tolist()


print(np.asarray(validation_images).shape)
print(np.asanyarray(validation_labels).shape)
print(np.asanyarray(validation_targets).shape)
print(height, width)

def create_model(train_images, train_targets, validation_images, validation_targets):
    #create the common input layer
    input_shape = (height, width, 3)
    input_layer = layers.Input(input_shape)

    #create the base layers
    base_layers = layers.Rescaling(1./255, name='base_1')(input_layer)
    base_layers = layers.Conv2D(16, 3, padding='same', activation='relu', name='base_2')(base_layers)
    base_layers = layers.MaxPooling2D(name='bl_3')(base_layers)
    base_layers = layers.Conv2D(32, 3, padding='same', activation='relu', name='base_4')(base_layers)
    base_layers = layers.MaxPooling2D(name='bl_5')(base_layers)
    base_layers = layers.Conv2D(64, 3, padding='same', activation='relu', name='base_6')(base_layers)
    base_layers = layers.MaxPooling2D(name='base_7')(base_layers)
    base_layers = layers.Flatten(name='base_8')(base_layers)

    #create the classifier branch
    classifier_branch = layers.Dense(128, activation='relu', name='class_1')(base_layers)
    classifier_branch = layers.Dense(num_classes, name='class_head')(classifier_branch)

    #create the localiser branch
    locator_branch = layers.Dense(128, activation='relu', name='local_1')(base_layers)
    locator_branch = layers.Dense(64, activation='relu', name='local_2')(locator_branch)
    locator_branch = layers.Dense(32, activation='relu', name='local_3')(locator_branch)
    locator_branch = layers.Dense(4, activation='sigmoid', name='local_head')(locator_branch)

    model = keras.Model(
        inputs = input_layer, 
        outputs = [classifier_branch,locator_branch]
        )
    
    model.compile(
        optimizer='Adam',
        loss=losses,  
        metrics=['accuracy']
    )

    model.fit(
        x=train_images,
        y=train_targets,
        batch_size=10,
        epochs=1,
        verbose='auto',
        callbacks=None,
        validation_split=0.0,
        validation_data=(validation_images, validation_targets),
        shuffle=True,
        class_weight=None,
        sample_weight=None,
        initial_epoch=0,
        steps_per_epoch=None,
        validation_steps=None,
        validation_batch_size=5,
        validation_freq=1,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=True
    )
    model.save("./my_models/duck_model.h5py")
    return model

model = create_model(train_images, train_targets, validation_images, validation_targets)


trainTargets = (train_labels, train_targets)

validationTargets = (validation_labels, validation_targets)

# train_labels = np.asarray(train_labels, dtype=np.int64)
# train_targets = np.asarray(train_targets, dtype=np.int64)

# trainTargets = np.asarray(trainTargets, dtype=np.int64)
# trainTargets = np.expand_dims(trainTargets, axis=-1)

# validationTargets = np.asarray(validationTargets, dtype=np.int64)
# validationTargets = np.expand_dims(validationTargets, axis=-1)

# validation_labels = np.asarray(validation_labels, dtype=np.int64)
# validation_targets = np.asarray(validation_targets, dtype=np.int64)

# train_images = np.array(train_images)
# validation_images = np.array(validation_images)









def last_attempt(current_frame):
    if os.path.exists("./my_models/duck_model.h5py") == False:
        model = create_model(train_images, train_targets, validation_images, validation_targets)
    else:
        model = keras.models.load_model("./my_models/duck_model.h5py")
    predictions = model.predict(
        current_frame,
        batch_size=None,
        verbose=0,
        steps=None,
        callbacks=None,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=True
    )
    print(predictions)
    xmin, ymin, xmax, ymax = predictions
    x = int((xmin+xmax)*(width/2))
    y = int((ymin+ymax)*(height/2))
    coordinates = [x, y]
    return coordinates





    
