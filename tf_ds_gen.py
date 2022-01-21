import os
import cv2
import numpy as np
import tensorflow as tf

from constants import BATCH_SIZE
from constants import EXTRA_PATHS
from constants import IMAGE_SHAPE
from constants import MASK_SHAPE
from constants import SHUFFLE_BUFFER
from constants import SLICE_NAMES
from constants import TEST_SPLIT
from constants import TRAIN_SPLIT
from constants import VAL_SPLIT
from constants import X_PATH
from constants import Y_PATH


def _data_generator():
    data_folders = []
    data_folders.append(X_PATH)
    data_folders += EXTRA_PATHS
    data_folders.append(Y_PATH)

    images = []
    with open(SLICE_NAMES) as f:
        for line in f:
            images.append(line.strip())
    
    if len(data_folders) > 2:
        for image in images:
            tmp = []
            tmp.append(cv2.imread('/'.join([data_folders[0], image])))
            for input_data_path in data_folders[1:-1]:
                tmp.append(cv2.imread('/'.join([input_data_path, image]), cv2.IMREAD_GRAYSCALE))
            yield (
                np.dstack(tmp)/255,
                cv2.imread('/'.join([data_folders[-1], image]), cv2.IMREAD_GRAYSCALE).reshape(MASK_SHAPE)/255
            )
    elif len(data_folders) == 2:
        for image in images:
            yield (
                cv2.imread('/'.join([data_folders[0], image]))/255,
                cv2.imread('/'.join([data_folders[-1], image]), cv2.IMREAD_GRAYSCALE).reshape(MASK_SHAPE)/255
            )
    else:
        print('Incorrect model input data paths')

def _gen_training_ds() -> tf.data.Dataset:
    return tf.data.Dataset.from_generator(
        _data_generator,
        output_signature=(
            tf.TensorSpec(
                shape=IMAGE_SHAPE,
                dtype=tf.float32
            ),
            tf.TensorSpec(
                shape=MASK_SHAPE,
                dtype=tf.float32
            )
        )
    )

def _get_dataset_partitions_tf(
    ds,
    ds_size,
    train_split=TRAIN_SPLIT,
    val_split=VAL_SPLIT,
    test_split=TEST_SPLIT,
    shuffle=True,
    shuffle_size=SHUFFLE_BUFFER) -> tuple:

    assert (train_split + test_split + val_split) == 1
    
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=42)
    
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    
    train_ds = ds.take(train_size)    
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    
    return train_ds, val_ds, test_ds

def load_training_data() -> tuple:
    ds_size = len(os.listdir(X_PATH))
    ds = _gen_training_ds()
    train, val, test = _get_dataset_partitions_tf(ds, ds_size)
    train = train.padded_batch(BATCH_SIZE)
    val = val.padded_batch(BATCH_SIZE)
    test = test.padded_batch(BATCH_SIZE)
    return train, val, test
