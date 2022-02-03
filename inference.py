import os

import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

from image_slicing_helper import extract_file_names_from_input_data

from constants import EXTRA_PATHS
from constants import IMAGE_SHAPE
from constants import INFERENCE_DATA_PATH
from constants import INFERENCE_OUTPUT
from constants import MODEL_NAME
from constants import MODEL_SAVE_FOLDER
from constants import ORTHO_NAMES
from constants import OUTPUT_MASK_RGB_COLOR
from constants import THRESHOLD_VALUE
from constants import X_PATH


def get_concat_h_multi_resize(im_list, resample=Image.BICUBIC):
    min_height = min(im.height for im in im_list)
    im_list_resize = [im.resize((int(im.width * min_height / im.height), min_height),resample=resample)
                      for im in im_list]
    total_width = sum(im.width for im in im_list_resize)
    dst = Image.new('RGB', (total_width, min_height))
    pos_x = 0
    for im in im_list_resize:
        dst.paste(im, (pos_x, 0))
        pos_x += im.width
    return dst

def get_concat_v_multi_resize(im_list, resample=Image.BICUBIC):
    min_width = min(im.width for im in im_list)
    im_list_resize = [im.resize((min_width, int(im.height * min_width / im.width)),resample=resample)
                      for im in im_list]
    total_height = sum(im.height for im in im_list_resize)
    dst = Image.new('RGB', (min_width, total_height))
    pos_y = 0
    for im in im_list_resize:
        dst.paste(im, (0, pos_y))
        pos_y += im.height
    return dst

def get_concat_tile_resize(im_list_2d, resample=Image.BICUBIC):
    im_list_v = [get_concat_h_multi_resize(im_list_h, resample=resample) for im_list_h in im_list_2d]
    return get_concat_v_multi_resize(im_list_v, resample=resample)

def data_generator(image_name):
    data_folders = []
    data_folders.append(f'{X_PATH[2:]}/{image_name}')
    for path in EXTRA_PATHS:
        data_folders.append(f'{path}/{image_name}')
    
    if len(data_folders) > 1:
        tmp = []
        img = cv2.imread(data_folders[0])
        tmp.append(img)
        for i in range(1, len(data_folders)):
            img = cv2.imread(data_folders[i], cv2.IMREAD_GRAYSCALE)
            tmp.append(img)
        return np.dstack(tmp)/255
    elif len(data_folders) == 1:
        return cv2.imread(data_folders[0])/255
    else:
        print('Missing model input data paths')


ensemble_model = tf.keras.models.load_model(f'{MODEL_SAVE_FOLDER}/{MODEL_NAME}.h5')

extract_file_names_from_input_data(X_PATH)

for image_name in sorted(os.listdir(X_PATH)):
    image = data_generator(image_name).reshape(1, 512, 512, IMAGE_SHAPE[-1])
    pred = ensemble_model.predict(image).reshape(512,512,1)
    threshold = THRESHOLD_VALUE
    pred[pred <= threshold] = 0.0
    pred[pred > threshold] = 1.0
    pred = np.stack(
        [
            pred * OUTPUT_MASK_RGB_COLOR[0], 
            pred * OUTPUT_MASK_RGB_COLOR[1], 
            pred * OUTPUT_MASK_RGB_COLOR[2], 
            pred
        ],
        axis=2).reshape(512, 512, 4)
    if not cv2.imwrite(f'{INFERENCE_DATA_PATH}/{image_name}', pred*255):
        print('problem')

names = []
with open(ORTHO_NAMES, 'r') as f:
    for name in f:
        tmp = []
        splt = name.strip().split('_')
        i = int(splt[-2])
        j = int(splt[-1])
        size_length = len(f'_{i}_{j}')+1
        tmp.append(name[:-size_length])
        tmp.append(i)
        tmp.append(j)
        names.append(tmp)

for name in names:
    images = []
    for i in range(name[1]):
        images.append([])
        for j in range(name[2]):
            images[i].append(Image.open(f'{INFERENCE_DATA_PATH}/{name[0]}_{i}_{j}.png'))

    get_concat_tile_resize(images).save(f'{INFERENCE_OUTPUT}/{name[0]}_mask.tif')
