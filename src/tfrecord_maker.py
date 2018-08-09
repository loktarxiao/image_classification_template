#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 20:01:00 2018

@author: loktar
"""

import sys
sys.dont_write_bytecode = True

import tensorflow as tf
from tqdm import tqdm
from multiprocessing import Pool
import numpy as np
from glob import glob
import cv2
import random
import os
import argparse


def _int64_feature(value):
    """Conver integer data to a string which is accepted by tensorflow record.
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    """Conver byte data to a string which is accepted by tensorflow record.
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def tfrecord_string(image, label):
    """Convert image array and label message to tensorflow serialized string.
    Args:
        image(numpy.array): image array read by opencv/scipy.misc/skimage.io/PIL.Image
                            in RGB order.
        label(int): the class of `image` which starts with `0`.

    Returns:
        tf_serialized: tensorflow serialized string message matched by `TFRECORD`.
    """

    feature = {
        'label': _int64_feature([label]),
        'image': _bytes_feature([image.tobytes()]),
        'image_shape': _int64_feature(list(image.shape))
    }
    tf_features = tf.train.Features(feature=feature)
    example = tf.train.Example(features=tf_features)
    tf_serialized = example.SerializeToString()
    return tf_serialized


def RecorderMessage(item):
    """Read image and convert it to serialized string.
    Args:
        item(list): the message list format is [`image_path`, `label`].
                    e.g. ['/data/dogs_vs_cats/train/dogs/001.jpg', 1]

    Returns:
        tf_serialized_string: The item tfrecord string.
    """
    image_path, label = item[0], item[1]
    assert isinstance(image_path, str)
    assert isinstance(label, int)

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    tf_serialized_string = tfrecord_string(image, label)
    print("Get the {} serialized string.".format(image_path))

    return tf_serialized_string


def CreatePathMessageFromFolder(target_folder,
                                shuffle=True,
                                image_types=['png', 'jpg', 'jpeg'],
                                balance=True):
    """Create item list of image paths
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

        shuffle(bool): Whether to shuffle image path list.

        image_types(list): The extension types of images.

        balance(bool): Whether to balance each class.


    Returns:
        images_path_message(list): the message list format is [[`image_path`, `label`], ...].
            e.g. [['/data/dogs_vs_cats/train/dogs/001.jpg', 1],
                    ['/data/dogs_vs_cats/train/dogs/001.jpg', 2],
                    ...]
    """
    class_path_lst = glob(os.path.join(target_folder, '*'))
    class_dict = {}
    image_path_message = []
    image_path_lst = []
    for index, class_path in enumerate(class_path_lst):
        if not os.path.isdir(class_path):
            continue
        class_name = class_path.split('/')[-1]
        class_dict[class_name] = index

        imagelist = []
        for image_type in image_types:
            imagelist += glob(os.path.join(class_path, '*'+image_type))

        image_path_lst.append([[path, index] for path in imagelist])

    if balance:
        max_num = max([len(i) for i in image_path_lst])
        for image_paths in image_path_lst:
            add_num = max_num - len(image_paths)
            add_index = list(np.random.choice(len(image_paths), add_num))
            image_paths += [image_paths[j] for j in add_index]

            image_path_message += image_paths
    else:
        for image_paths in image_path_lst:
            image_path_message += image_paths

    if shuffle:
        random.shuffle(image_path_message)
    return image_path_message


def RecordMaker(item_lst, writer_path, num_processes=5):
    """Write images messages to `TFRECORD`
    Args:
        item_lst(list): image paths list, and the format is [[`image_path`, `label`], ...].
            e.g. [['/data/dogs_vs_cats/train/dogs/001.jpg', 1],
                    ['/data/dogs_vs_cats/train/dogs/001.jpg', 2],
                    ...]

        writer_path(string): the output path of `TFRECORD`

        num_processes(int): The number of processors

    Returns:
       None 
    """
    writer = tf.python_io.TFRecordWriter(writer_path)

    from multiprocessing import Pool
    pool = Pool(processes=num_processes)
    results = []

    for item in item_lst[:2000]:
        results.append(pool.apply_async(RecorderMessage, args=(item,)))

    pool.close()
    pool.join()

    print("\n[INFO] Tasks has been distributed to each pool.")
    print("[WAITING] ......\n")
    for result in tqdm(results):
        writer.write(result.get())

    print("\n[INFO] Tasks has been completed.\n")

    writer.close()


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Create tensorflow record files \
        to make the project database')

    parser.add_argument('--data-folder',
                        help='The dataset path')

    parser.add_argument('--name', default='untitled',
                        help='The tfrecord dataset name')

    parser.add_argument('--outpath', default='../output/tfrecord',
                        help='The direction of tfrecord output path.')

    parser.add_argument('--balance', type=bool, default=True,
                        help='Need to balance positive and negative samples or not.')

    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='The ratio of train set and valid set.')

    parser.add_argument('--num-processors', type=float, default=8,
                        help='The ratio of train set and valid set.')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    output_folder = os.path.join(args.outpath, args.name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    train_folder = args.data_folder

    train_valid_item_lst = CreatePathMessageFromFolder(
        train_folder, balance=True, shuffle=True)
    split_number = int(args.train_ratio * len(train_valid_item_lst))
    train_lst = train_valid_item_lst[:split_number]
    valid_lst = train_valid_item_lst[split_number:]

    train_record_path = os.path.join(output_folder, 'train.tfrecord')
    valid_record_path = os.path.join(output_folder, 'valid.tfrecord')
    RecordMaker(train_lst, train_record_path)
    RecordMaker(valid_lst, valid_record_path)
