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
import tqdm

from constants import SLICE_NAMES


def extract_file_names_from_input_data(x_path: str):
    with open(SLICE_NAMES, 'w') as f:
        for image in sorted(os.listdir(x_path)):
            f.write(image + '\n')


def slice_image(
    image: np.ndarray,
    slice_size: int,
    slice_prefix: str,
    offset_h_w: tuple = (0, 0)
    ) -> dict:

    out = dict()
    h, w = image.shape[:2]
    for slice_y in range(int((h-offset_h_w[0])/slice_size)):
        for slice_x in range(int((w-offset_h_w[1])/slice_size)):
            out['{}_{}_{}'.format(slice_prefix, slice_y, slice_x)] = \
                image[offset_h_w[0]+slice_y*slice_size:
                      offset_h_w[0]+slice_y*slice_size+slice_size,
                      offset_h_w[1]+slice_x*slice_size:
                      offset_h_w[1]+slice_x*slice_size+slice_size]
    return out


def generate_color_based_heatmap(
    original_images: str,
    output_path: str,
    ):

    print('Generating green filtered images')
    images = []
    with open(SLICE_NAMES) as f:
        for line in f:
            images.append(line.strip())
    
    for image in tqdm.tqdm(images):
        color_normal_image = cv2.imread('/'.join([original_images, image]))

        # Filtering based on majority of color (Green->0, Red/Blue->255)
        for x in range(len(color_normal_image)):
            for y in range(len(color_normal_image[x])):
                if color_normal_image[x, y, 1] >= color_normal_image[x, y, 0] \
                and color_normal_image[x, y, 1] >= color_normal_image[x, y, 2]:
                    color_normal_image[x, y] = 0
                else:
                    color_normal_image[x, y] = 255

        # Blur heatmap to become a heatmap instead of a mask
        color_normal_image = cv2.GaussianBlur(color_normal_image, (99, 99), cv2.BORDER_DEFAULT)
                
        if not cv2.imwrite('/'.join([output_path, image]), color_normal_image):
            raise Exception("Could not write image")
