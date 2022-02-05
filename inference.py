''' AI Support for building detection
    Copyright (C) 2022  Zsombor Toth

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
'''


import os

import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

from image_slicing_helper import extract_file_names_from_input_data
from pillow_concat import get_concat_tile_resize
from ortho_slicing import slice_images

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


ortho_size, _ = slice_images()
ensemble_model = tf.keras.models.load_model(f'{MODEL_SAVE_FOLDER}/{MODEL_NAME}.h5')
extract_file_names_from_input_data(X_PATH)

# Importing preprocessed image slices and predicting their respective masks
for image_name in sorted(os.listdir(X_PATH)):
    image = data_generator(image_name).reshape(1, IMAGE_SHAPE[0], IMAGE_SHAPE[1], IMAGE_SHAPE[2])
    pred = ensemble_model.predict(image).reshape(IMAGE_SHAPE[0], IMAGE_SHAPE[1], 1)
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
        axis=2).reshape(IMAGE_SHAPE[0], IMAGE_SHAPE[1], 4)
    if not cv2.imwrite(f'{INFERENCE_DATA_PATH}/{image_name}', pred*255):
        print('problem')

# Collecting metainformation for output reconstruction
# The names of the last slices are saved when slicing the input images so and
# since they contain the number of sliced rows and columns, these names can
# help to collect the correct slices and the correct number of slices per
# orthophoto
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

# Reconstructing and saving the output mask from the predicted slices
for name in names:
    # Collecting predicted slices
    images = []
    for i in range(name[1] + 1):
        images.append([])
        for j in range(name[2] + 1):
            images[i].append(Image.open(f'{INFERENCE_DATA_PATH}/{name[0]}_{i}_{j}.png'))
    # Reconstructing image from predicted slices
    output = get_concat_tile_resize(images)
    # Removing padding from the edges
    output = output.crop((0, 0, ortho_size[f'{name[0]}.tif'][1], ortho_size[f'{name[0]}.tif'][0]))
    # Saving output image
    output.save(f'{INFERENCE_OUTPUT}/{name[0]}_mask.tif')
