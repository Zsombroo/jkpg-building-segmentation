# TODO: DOCUMENT IT: BOTH MASK AND ORTHOPOTO HAVE TO BE NAMED THE SAME


import glob
import os

import cv2
import tqdm

from image_slicing_helper import extract_file_names_from_input_data
from image_slicing_helper import generate_color_based_heatmap
from image_slicing_helper import slice_image

from constants import GREEN_FILTER_PATH
from constants import EXTRA_PATHS
from constants import INFERENCE_DATA_PATH
from constants import INFERENCE_OUTPUT
from constants import ORTHOPHOTOS
from constants import ORTHOPHOTO_MASKS
from constants import ORTHO_NAMES
from constants import X_PATH
from constants import Y_PATH


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
folders_to_emppty = [GREEN_FILTER_PATH, INFERENCE_DATA_PATH, X_PATH, Y_PATH]
for folder in folders_to_emppty:
    files = glob.glob(f'{folder}/*')
    for f in files:
        os.remove(f)

ortho_last_slice_names = []
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
        ortho_last_slice_names.append(k)

mask_last_slice_names = []
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
