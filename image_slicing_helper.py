import os
import cv2
import numpy as np
import tqdm

from constants import SLICE_NAMES


def extract_file_names_from_input_data(x_path: str):
    with open(SLICE_NAMES, 'w') as f:
        for image in sorted(os.listdir(x_path)):
            f.write(image + '\n')


def normalize(x):
    mag = np.sqrt(x.dot(x))
    if mag != 0:
        return x/mag
    else:
        return x


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

        # Per pixel color vector normalization
        for x in range(len(color_normal_image)):
            for y in range(len(color_normal_image[x])):
                color_normal_image[x, y] = normalize(color_normal_image[x, y])

        # Filtering based on majority of color (Green->0, Red/Blue->255)
        image_heatmap = np.ndarray((512, 512))
        for x in range(len(image_heatmap)):
            for y in range(len(image_heatmap[x])):
                if color_normal_image[x, y, 1] > color_normal_image[x, y, 0] \
                and color_normal_image[x, y, 1] > color_normal_image[x, y, 2]:
                    image_heatmap[x, y] = 0
                else:
                    image_heatmap[x, y] = 255

        # Blur heatmap to become a heatmap instead of a mask
        for i in range(1):
            image_heatmap = cv2.GaussianBlur(
                image_heatmap, 
                (99, 99), 
                cv2.BORDER_DEFAULT,
            )
                
        if not cv2.imwrite('/'.join([output_path, image]), image_heatmap):
            raise Exception("Could not write image")
