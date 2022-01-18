import os

import cv2
import tqdm

from image_slicing_helper import extract_file_names_from_input_data
from image_slicing_helper import generate_color_based_heatmap
from image_slicing_helper import slice_image

from constants import GREEN_FILTER_PATH
from constants import ORTHOPHOTOS
from constants import ORTHOPHOTO_MASKS
from constants import X_PATH
from constants import Y_PATH


# TODO: BOTH MASK AND ORTHOPOTO HAVE TO BE NAMED THE SAME

# Slice orthophotos
if os.listdir(ORTHOPHOTOS) != 0:
    for image in os.listdir(ORTHOPHOTOS):
        print(f'Slicing {ORTHOPHOTOS}/{image}')
        slices = slice_image(
            cv2.imread('/'.join((ORTHOPHOTOS, image))),
            512,
            image.split('.')[0],
        )
        for k, v in tqdm.tqdm(slices.items()):
            cv2.imwrite(f'{X_PATH}/{k}.png', v)
# Slice orthophoto masks
if os.listdir(ORTHOPHOTO_MASKS) != 0:
    for image in os.listdir(ORTHOPHOTO_MASKS):
        print(f'Slicing {ORTHOPHOTO_MASKS}/{image}')
        slices = slice_image(
            cv2.imread('/'.join((ORTHOPHOTO_MASKS, image))),
            512,
            image.split('.')[0],
        )
        for k, v in tqdm.tqdm(slices.items()):
            cv2.imwrite(f'{Y_PATH}/{k}.png', v)

extract_file_names_from_input_data(X_PATH)

generate_color_based_heatmap(X_PATH, GREEN_FILTER_PATH)
