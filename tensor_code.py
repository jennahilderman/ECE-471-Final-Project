import os
from cv2 import threshold
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import keras
from keras import layers
from keras.models import Sequential
import keras.optimizers
import keras.metrics
import keras.losses
import torch.nn as nn
import torch.nn.functional as F
import datetime


from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils 
from object_detection.builders import model_builder
from object_detection.utils import shape_utils
from tensorflow.python.ops.numpy_ops import np_config

red_path = "./duck_images/duck_images_red/"
red_name = "red_"
black_path = "./duck_images/duck_images_black/"
black_name = "black_"
blue_path = "./duck_images/duck_images_blue/"
blue_name = "blue_"
dead_path = "./duck_images/dead_duck_images/"
dead_name = "dead_"
label_map_path = "./label_map.pbtxt"

red_label = 1
black_label = 2
blue_label = 3
dead_label = 4
done_training = False

test_data_map = "duck_images\\duck_data\\test\\duck_label_map.pbtxt"
test_data_set = "duck_images\\duck_data\\test\\duck.tfrecord"
#Recover saved model\

ckpt = tf.train.Checkpoint(v=tf.Variable(1.))
path = ckpt.write('/training_demo/models/checkpoints')

config_file_path = "training_demo\\models\\config_file.config"
check_point_path = "training_demo\\duck_data\\train\\my_train_duck_model\\ckpt-5"



configs = config_util.get_configs_from_pipeline_file(config_file_path)
model_config = configs['model']
detection_model = model_builder.build(
    model_config=model_config, is_training=False
)
check_point = tf.compat.v2.train.Checkpoint(
    model=detection_model
)
check_point.restore(os.path.join(check_point_path)).expect_partial()


label_map = label_map_util.load_labelmap(label_map_path)
categories = label_map_util.convert_label_map_to_categories(
    label_map,
    max_num_classes=label_map_util.get_max_label_map_index(label_map),
    use_display_name=True
    )
category_index = label_map_util.create_category_index(categories)
label_map_dict = label_map_util.get_label_map_dict(label_map, use_display_name=True)



def load_image(file_path, file_name):
    image_array = []
    for i in range(18):
        img = cv2.imread(file_path + file_name + str(i+1) + ".png")
        print(file_path + file_name + str(i+1) + ".png")
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if img is not None:
            image_array.append(img)        
    return image_array


def create_labels(image_array, label):
    label_array = []
    for i in range(len(image_array)):
        label_array.append(label)
    return label_array

def merge(image_array_1, label_array_1, image_array_2, label_array_2):
    merged_image_array = np.concatenate((image_array_1, image_array_2))
    #merged_image_array = tf.concat(image_array_1, image_array_2, 0)
    merged_label_array = np.append(label_array_1, label_array_2)
    #print(merged_label_array)
    #merged_label_array = tf.concat(label_array_1, label_array_2, 0)
    return merged_image_array, merged_label_array

def split_array(image_array, label_array, length):
    rand_array = np.random.permutation(len(label_array))
    test_images = []
    test_labels = []
    train_images = []
    train_labels = []
    red, black, blue, dead = 0
    j = 0
    for i in len(rand_array):
        if(label_array[i] == "red" and red <= length/4):
            test_images.append(image_array[rand_array[i]])
            test_labels.append(label_array[rand_array[i]])
            red += 1
        elif(label_array[i] == "black" and black <= length/4):
            test_images.append(image_array[rand_array[i]])
            test_labels.append(label_array[rand_array[i]])
            black += 1
        elif(label_array[i+j] == "blue" and blue <= length/4):
            test_images.append(image_array[rand_array[i]])
            test_labels.append(label_array[rand_array[i]])
            blue += 1
        elif(label_array[i+j] == "dead" and dead <= length/4):
            test_images.append(image_array[rand_array[i]])
            test_labels.append(label_array[rand_array[i]])
            dead += 1
        else:
            train_images.append(image_array[rand_array[i]])
            train_labels.append(label_array[rand_array[i]])
    return ((test_images, test_labels), (train_images, train_labels))


def convert_to_tf(image):
    print("convert to tf")
    image = tf.constant(image)
    image = image[tf.newaxis,...]
    image.shape.as_list()
    tf.image.resize(image, [3,5])[0,...,0].numpy()
    return (image)

def model_creation(data, labels):
    model = keras.Sequential()
    # Convolutions
    model.add(tf.keras.Input(shape=(64,64,3)))
    model.add(layers.Conv2D(32, kernel_size = (3,3), activation = 'relu', padding = 'same'))
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D(pool_size = (2,2), padding = 'same'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Conv2D(64, kernel_size = (3,3), activation = 'relu', padding = 'same'))
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D(pool_size = (2,2), padding = 'same'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(32, kernel_size = (3,3), activation = 'relu', padding = 'same'))
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D(pool_size = (2,2), padding = 'same'))
    model.add(layers.Dropout(0.3))
    # FCNN
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation = 'relu'))
    model.add(layers.ReLU())
    # Into 4 classes
    model.add(layers.Dense(4, activation = 'softmax'))

    model.compile(
        optimizer = 'rmsprop',
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy']
    )

    model.fit(
        data,
        labels,
        batch_size = 32,
        epochs=1, verbose=2,
        callbacks=None,
        validation_split=0.2,
        shuffle = True,
        class_weight=None,
        sample_weight=None,
        initial_epoch=0
    )

    model.train_on_batch(
        data,
        labels,
        class_weight=None,
        sample_weight=None,
    )

    # model.fit_generator(
    #     data,
    #     labels,
    #     steps_per_epoch=900,
    #     epochs=1,
    #     verbose=2,
    #     callbacks=None,
    #     validation_data=None,
    #     validation_steps=None,
    #     class_weight=None,
    #     max_queue_size=10,
    #     workers=1,
    #     initial_epoch=0
    # )

    model.evaluate(
        data,
        labels,
        batch_size=32,
        verbose=1,
        sample_weight=None   
    )

    model.predict(
        data,
        batch_size=32,
        verbose=2
    )
    trained_model = model.save("./my_models/duck_model.h5py")
    return trained_model

# def model_creation(data, labels):
#     model = keras.Sequential(
#         [
#             # Convolutions
#             layers.Conv2D(32, kernel_size = (3,3), activation = 'relu', padding = 'same'),
#             layers.ReLU(),
#             layers.MaxPooling2D(pool_size = (2,2), padding = 'same'),
#             layers.Dropout(0.1),
#             layers.Conv2D(64, kernel_size = (3,3), activation = 'relu', padding = 'same'),
#             layers.ReLU(),
#             layers.MaxPooling2D(pool_size = (2,2), padding = 'same'),
#             layers.Dropout(0.2),
#             layers.Conv2D(32, kernel_size = (3,3), activation = 'relu', padding = 'same'),
#             layers.ReLU(),
#             layers.MaxPooling2D(pool_size = (2,2), padding = 'same'),
#             layers.Dropout(0.3),
#             # FCNN
#             layers.Flatten(),
#             layers.Dense(128, activation = 'relu'),
#             layers.ReLU(),
#             # Into 4 classes
#             layers.Dense(4, activation = 'softmax')
#         ],
#         name = 'model',
#     )

#     model.compile(
#         optimizer = 'rmsprop',
#         loss = 'sparse_categorical_crossentropy',
#         metrics = ['accuracy']
#     )

#     model.fit(
#         data,
#         labels,
#         batch_size=32,
#         epochs=1, verbose=2,
#         callbacks=None,
#         validation_split=0.2,
#         shuffle = True,
#         class_weight=None,
#         sample_weight=None,
#         initial_epoch=0
#     )

#     model.train_on_batch(
#         data,
#         labels,
#         class_weight=None,
#         sample_weight=None,
#     )

#     # model.fit_generator(
#     #     data,
#     #     labels,
#     #     steps_per_epoch=900,
#     #     epochs=1,
#     #     verbose=2,
#     #     callbacks=None,
#     #     validation_data=None,
#     #     validation_steps=None,
#     #     class_weight=None,
#     #     max_queue_size=10,
#     #     workers=1,
#     #     initial_epoch=0
#     # )

#     model.evaluate(
#         data,
#         labels,
#         batch_size=32,
#         verbose=1,
#         sample_weight=None   
#     )

#     model.predict(
#         data,
#         batch_size=32,
#         verbose=2
#     )
#     trained_model = model.save("./my_models/duck_model.h5py")
#     return trained_model



def get_model_detection_function(model):
  """Get a tf.function for detection."""

  @tf.function
  def detection_function(image):
    """Detect objects in image."""
    image, shapes = model.preprocess(image)
    prediction_dict = model.predict(image, shapes)
    detections = model.postprocess(prediction_dict, shapes)

    return detections

  return detection_function

detection_function = get_model_detection_function(detection_model)


def detect(images, model):
    print("detect\n")
    for single_image in images:
        single_image = convert_to_tf(single_image)
        # image_np = load_image_into_numpy_array(image_path)            
        input_tensor = tf.convert_to_tensor(
            np.expand_dims(single_image, 0), dtype=tf.float32)
        detections = detection_function(input_tensor)
        visualization_utils.visualize_boxes_and_labels_on_image_array(
              single_image,
              detections['detection_boxes'][0].numpy(),
              detections['detection_classes'][0].numpy(),
              detections['detection_scores'][0].numpy(),
              category_index,
              use_normalized_coordinates = True,
              max_boxes_to_draw = 15,
              min_score_thresh = .5,
              agnostic_mode = False,
              line_thickness = 4,
              skip_scores = False,
              skip_labels= False  
        )
        scores = detections['detection_scores'][0]
        score_threshold = 0.5
        coordinates = []
        for i in range(len(scores)):
            print( i )
            print(detections['detection_classes'])
            # print(detections['detection_classes'][i])
            class_id = detections['detection_classes'][i]
            if scores[i] > score_threshold and class_id[i].astype(int) + 1 < 4:
                y_min = single_image[0]
                x_min = single_image[1]
                y_max = single_image[2]
                x_max = single_image[3]
                x = (x_max+x_min)*single_image.shape[1]/2
                y = (y_max+y_min)*single_image.shape[0]/2
                coordinates = [x,y]
                print(coordinates)
                return coordinates
    return

def train_the_model(red_path, red_name, black_path, black_name, blue_path, blue_name, dead_path, dead_name):
    red_ducks = load_image(red_path, red_name)
    red_labels = create_labels(red_ducks, red_label)
    black_ducks = load_image(black_path, black_name)
    black_labels = create_labels(black_ducks, black_label)
    red_black_ducks, red_black_labels = merge(convert_to_tf(red_ducks), red_labels, convert_to_tf(black_ducks), black_labels)
    blue_ducks = load_image(blue_path, blue_name)
    blue_labels = create_labels(blue_ducks, blue_label)
    red_black_blue_ducks, red_black_blue_labels = merge(red_black_ducks, red_black_labels, blue_ducks, blue_labels)
    dead_ducks = load_image(dead_path, dead_name)
    dead_labels = create_labels(dead_ducks, dead_label)
    all_ducks, all_labels = merge(red_black_blue_ducks, red_black_blue_labels, dead_ducks, dead_labels)
    print("ALL DUCKS MERGED****************************************************")
    ducks_norm = convert_to_tf(all_ducks)
    print("CONVERTED IMAGES****************************************************")
    model = model_creation(ducks_norm, all_labels)
    return model

def tensor(current_frame):
    coordinate = detect(current_frame, detection_model)
    return coordinate
if __name__ == "__main__":
    tensor(True)



    
    
