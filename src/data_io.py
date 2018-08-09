#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 23:52:46 2018

@author: loktar
"""

import sys
import os
sys.dont_write_bytecode = True

import tensorflow as tf
from tfrecord_maker import CreatePathMessageFromFolder
from data_preprocessing import preprocessing_func, augmentation_func, resize_output_image


def _parse_function(example_proto):
    """Resolve the TFRECORD string message to tensors.
    Args:
        example_proto: tensorflow proto string handle. Get it from `tf.data.TFRecordDataset`

    Returns:
        The data results of resolving. [image, label] 
    
    """
    dics = {
        'image': tf.FixedLenFeature([], dtype=tf.string),
        'image_shape': tf.FixedLenFeature(shape=(3, ), dtype=tf.int64),
        'label': tf.FixedLenFeature([], dtype=tf.int64),
    }
    parsed_example = tf.parse_single_example(example_proto, features=dics)

    image = tf.reshape(tf.decode_raw(
        parsed_example['image'], tf.uint8), parsed_example['image_shape'])
    label = parsed_example['label']

    image = tf.cast(image, tf.float32)
    label = tf.cast(label, tf.float32)

    return image, label


def _ImageRead_function(filename, label):
    """Read images from file path via tensorflow API.
    Args:
        filename(str): the image path.
        label(int or float): label message.
    
    Return:
        image, label: image tensor and label tensor.
    """
    image_string = tf.read_file(filename)
    image = tf.image.decode_image(image_string)
    return image, label


def dataset_from_tfrcord(tfrecord_lst, num_processors=8):
    """Create `tf.dataset` by tfrecord
    Args:
        tfrecord_lst(list): a list included all paths of tfrecord
        num_processors(int): number of processor to load data.
    """
    with tf.variable_scope("TFRECORD_DATASET"):
        dataset = tf.data.TFRecordDataset(tfrecord_lst)
        dataset = dataset.map(_parse_function, num_processors)

    return dataset


def dataset_from_folder(target_folder, num_processors=8):
    """Create `tf.dataset` from folder
    Args:
        target_folder(str): A folder that contains all classes of images,
            e.g.
            target_folder
            ├── class0
                ├── image0.jpg
                ├── image1.jpg
                ├── ...
            ├── class1
                ├── image256.jpg
                ├── ...
            ├── ...

        num_processors(int): number of processor to load data.
    """
    image_path_lst = CreatePathMessageFromFolder(target_folder)
    image_path_message = tf.constant([i[0] for i in image_path_lst])
    label_message = tf.constant([i[1] for i in image_path_lst])
    with tf.variable_scope("IMAGEFOLDER_DATASET"):
        dataset = tf.data.Dataset.from_tensor_slices(
            (image_path_message, label_message))
        dataset = dataset.map(_ImageRead_function, num_processors)

    return dataset


def data_loader(input_handle,
                mode='tfrecord',
                num_repeat=1,
                shuffle=False,
                batch_size=128,
                num_processors=4,
                buffer_size=1000,
                augmentation=False,
                name="Untitle",
                device='cpu:0'):
    
    """ Creat an Iterator to load data
    Args:
        input_handle(str or list): the input handle to receive loading message.
        
        mode(str): a switch controlling input type
            if `mode` is 'tfrecord', `input handle` must be a tensorflow record list.
            if `mode` is 'folder', `input handle` must be a image folder like `target_folder`
        
        num_repeat(int): the number of dataset repeating times.
        
        batch_size(int): the data number of a batch
        
        shuffle(bool): whether to shuffle data.

        num_processors(int): number of processor to load data.

        augmentation(bool): whether to apply augmentation to the loading data.

        name(str): the name scope of data loader

        device(str): which device to put the data loader.
            the option is `cpu:0`/`gpu:0`/`gpu:1`/`gpu:2`/...
    """

    with tf.variable_scope(name, reuse=False):
        with tf.device('/{}'.format(device)):
            if mode == 'tfrecord':
                dataset = dataset_from_tfrcord(input_handle, num_processors)
            elif mode == 'folder':
                dataset = dataset_from_folder(input_handle, num_processors)

            dataset = dataset.map(preprocessing_func, num_processors)

            if shuffle:
                dataset = dataset.shuffle(buffer_size=buffer_size)

            if augmentation:
                dataset = dataset.map(augmentation_func, num_processors)

            dataset = dataset.map(resize_output_image, num_processors)
            dataset = dataset.repeat(num_repeat)
            dataset = dataset.batch(batch_size)

            iterator = dataset.make_initializable_iterator()

    return iterator


if __name__ == '__main__':

    train_tfrecord_lst = ['../output/tfrecord/dogs_vs_cats/train.tfrecord']
    folder = '/data/dogs_vs_cats/train'
    train_loader = data_loader(folder,
                               mode='folder',
                               num_repeat=1,
                               shuffle=True,
                               batch_size=12,
                               num_processors=4,
                               augmentation=True,
                               name='train_dataloader')

    sess = tf.InteractiveSession()
    sess.run(train_loader.initializer)
    test_pipeline = train_loader.get_next()
    for i in range(22):
        try:
            image, label = sess.run(test_pipeline)
            print(sum(label), image.shape)
        except tf.errors.OutOfRangeError:
            print('OOR')
            sess.run(train_loader.initializer)
    writer = tf.summary.FileWriter('test', sess.graph)
    sess.close()
    writer.close()

    import matplotlib.pyplot as plt
    for i, z in enumerate(image):
        print label[i]
        plt.imshow(z)
        plt.show()
