import cv2
import numpy as np
import os
import PIL
import tensorflow as tf

red_path = "./duck_images/red_images/"
red_name = "red_"
black_path = "./duck_images/black_images/"
black_name = "black_"
blue_path = "./duck_images/blue_images/"
blue_name = "blue_"
dead_path = "./duck_images/dead_images/"
dead_name = "dead_"


def load_image(file_path, file_name):
    image_array = []
    for i in range(18):
        img = cv2.imread(file_path + file_name + str(i+1) + ".png")
        cv2.imshow('img', img)
        if img is not None:
             np.append(image_array, img)         
    return image_array


def create_labels(image_array, label):
    label_array = []
    for i in range(len(image_array)):
        label_array.append(label)
    return label_array

def merge(image_array_1, label_array_1, image_array_2, label_array_2):
    merged_image_array = np.concatenate(image_array_1, image_array_2)
    merged_label_array = np.concatenate(label_array_1, label_array_2)
    return merged_image_array, merged_label_array

def create_tf_array(image_array, label_array):
    if len(image_array) == len(label_array):
        duck_dataset = tf.data.Dataset.from_tensor_slices((image_array, label_array))
    else:
        print("Lengths don't match")
    return duck_dataset

def tensor_main():
    red_ducks = load_image(red_path, red_name)
    red_labels = create_labels(red_ducks, "red")
    black_ducks = load_image(black_path, black_name)
    black_labels = create_labels(black_ducks, "black")
    red_black_ducks, red_black_labels = merge(red_ducks, red_labels, black_ducks, black_labels)
    blue_ducks = load_image(blue_path, blue_name)
    blue_labels = create_labels(blue_ducks, "blue")
    red_black_blue_ducks, red_black_blue_labels = merge(red_black_ducks, red_black_labels, blue_ducks, blue_labels)
    dead_ducks = load_image(dead_path, dead_name)
    dead_labels = create_labels(dead_ducks, "dead")
    all_ducks, all_labels = merge(red_black_blue_ducks, red_black_blue_labels, dead_ducks, dead_labels)
    duck_dataset = create_tf_array(all_ducks, all_labels)

    return dataset
    
