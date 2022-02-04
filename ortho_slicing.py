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


import glob
import os

import cv2
import numpy as np
import tqdm

from image_slicing_helper import extract_file_names_from_input_data
from image_slicing_helper import generate_color_based_heatmap
from image_slicing_helper import slice_image

from constants import GREEN_FILTER_PATH, MASK_SHAPE
from constants import EXTRA_PATHS
from constants import INFERENCE_DATA_PATH
from constants import INFERENCE_OUTPUT
from constants import ORTHOPHOTOS
from constants import ORTHOPHOTO_MASKS
from constants import ORTHO_NAMES
from constants import X_PATH
from constants import Y_PATH


def slice_images():
    # Generate defined folders if they do not exist
    folders_to_check = [
        ORTHOPHOTOS,
        ORTHOPHOTO_MASKS,
        X_PATH,
        Y_PATH,
        INFERENCE_DATA_PATH,
        INFERENCE_OUTPUT
    ]
    for path in EXTRA_PATHS:
        folders_to_check.append(path)

    for folder in folders_to_check:
        if not os.path.exists(folder):
            print(f'Creating {folder}')
            os.makedirs(folder)

    # Remove files from previous run
    folders_to_emppty = [
        X_PATH, 
        Y_PATH,
        INFERENCE_DATA_PATH
    ]
    for path in EXTRA_PATHS:
        folders_to_emppty.append(path)

    for folder in folders_to_emppty:
        files = glob.glob(f'{folder}/*')
        for f in files:
            os.remove(f)

    ortho_size = {}
    mask_size = {}

    ortho_last_slice_names = []
    # Slice orthophotos
    if os.listdir(ORTHOPHOTOS) != 0:
        for image in os.listdir(ORTHOPHOTOS):
            print(f'Slicing {ORTHOPHOTOS}/{image}')
            # Read orthophoto
            img_array = cv2.imread('/'.join((ORTHOPHOTOS, image)))
            # Pad image to n*512
            h, w, _ = img_array.shape
            h_pad = 512 - (h%512)
            w_pad = 512 - (w%512)
            ortho_size[image] = (h, w)
            img_padded = np.pad(img_array, ((0, h_pad), (0, w_pad), (0, 0)))
            # Slice image
            slices = slice_image(
                img_padded,
                512,
                image.split('.')[0],
            )
            for k, v in tqdm.tqdm(slices.items()):
                cv2.imwrite(f'{X_PATH}/{k}.png', v)
            ortho_last_slice_names.append(k)

    mask_last_slice_names = []
    # Slice orthophoto masks
    if os.listdir(ORTHOPHOTO_MASKS) != 0:
        for image in os.listdir(ORTHOPHOTO_MASKS):
            print(f'Slicing {ORTHOPHOTO_MASKS}/{image}')
            # Read orthophoto mask
            img_array = cv2.imread('/'.join((ORTHOPHOTO_MASKS, image)))
            # Pad mask to n*512
            h, w, _ = img_array.shape
            h_pad = 512 - (h%512)
            w_pad = 512 - (w%512)
            mask_size[image] = (h_pad, w_pad)
            img_padded = np.pad(img_array, ((0, h_pad), (0, w_pad), (0, 0)))
            # Slice mask
            slices = slice_image(
                img_padded,
                512,
                image.split('.')[0],
            )
            for k, v in tqdm.tqdm(slices.items()):
                cv2.imwrite(f'{Y_PATH}/{k}.png', v)
            mask_last_slice_names.append(k)

    for file in mask_last_slice_names:
        if file not in ortho_last_slice_names:
            print(f'ERROR: Mask ({file.split("_")[0]}) has no corresponding orthophoto')
            exit()

    with open(ORTHO_NAMES, 'w') as f:
        for i in ortho_last_slice_names:
            f.write(f'{i}\n')

    extract_file_names_from_input_data(X_PATH)

    generate_color_based_heatmap(X_PATH, GREEN_FILTER_PATH)

    return ortho_size, mask_size