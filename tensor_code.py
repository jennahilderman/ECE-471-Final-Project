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

# Define label numbers
red_label = 1
black_label = 2
blue_label = 3
dead_label = 4
done_training = False

# test data paths
test_data_map = "duck_images\\duck_data\\test\\duck_label_map.pbtxt"
test_data_set = "duck_images\\duck_data\\test\\duck.tfrecord"

# Recover model from checkpoint
ckpt = tf.train.Checkpoint(v=tf.Variable(1.))
path = ckpt.write('/training_demo/models/checkpoints')
config_file_path = "training_demo\\models\\config_file.config"
check_point_path = "training_demo\\duck_data\\train\\my_train_duck_model\\ckpt-5"

# label map ppath
label_map_path = "./label_map.pbtxt"

# Get model
configs = config_util.get_configs_from_pipeline_file(config_file_path)
model_config = configs['model']
detection_model = model_builder.build(
    model_config=model_config, is_training=False
)
check_point = tf.compat.v2.train.Checkpoint(
    model=detection_model
)

# restore model
check_point.restore(os.path.join(check_point_path)).expect_partial()

# get label map
label_map = label_map_util.load_labelmap(label_map_path)
categories = label_map_util.convert_label_map_to_categories(
    label_map,
    max_num_classes=label_map_util.get_max_label_map_index(label_map),
    use_display_name=True
    )
category_index = label_map_util.create_category_index(categories)
label_map_dict = label_map_util.get_label_map_dict(label_map, use_display_name=True)

# function converts image to a tensor array
def convert_to_tf(image):
    print("convert to tf")
    image = tf.constant(image)
    image = image[tf.newaxis,...]
    image.shape.as_list()
    tf.image.resize(image, [3,5])[0,...,0].numpy()
    return (image)


# Make a detection function using the model
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


def detect(images):
    print("detect\n")
    for single_image in images:

        single_image = convert_to_tf(single_image)
        # image_np = load_image_into_numpy_array(image_path)            
        input_tensor = tf.convert_to_tensor(
            np.expand_dims(single_image, 0), dtype=tf.float32
        )
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
              line_thickness = 0,
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




def tensor(current_frame):
    coordinate = detect(current_frame)
    return coordinate


if __name__ == "__main__":
    tensor(True)



    
    
