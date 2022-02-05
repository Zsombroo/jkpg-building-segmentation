###
### Folder structure
###
ORTHOPHOTOS = './data/ortho'
ORTHOPHOTO_MASKS = './data/ortho_mask'
INFERENCE_OUTPUT = './data/inference_output'

SLICE_NAMES = './data/names.txt'
ORTHO_NAMES = './data/ortho_names.txt'

MODEL_SAVE_FOLDER = './models'
INFERENCE_DATA_PATH = './data/inference_data'
X_PATH = './data/training_raw_data'
Y_PATH = './data/training_label_data'
GREEN_FILTER_PATH = './data/green_filter'
EXTRA_PATHS = [
    GREEN_FILTER_PATH,
]



###
### Model parameters
###

MODEL_NAME = 'green_filtered_model'  # Name of the ensemble model

IMAGE_SIDE_LENGTH = 512
IMAGE_SHAPE = (IMAGE_SIDE_LENGTH, IMAGE_SIDE_LENGTH, 3+len(EXTRA_PATHS))
MASK_SHAPE = (IMAGE_SIDE_LENGTH, IMAGE_SIDE_LENGTH, 1)

ENSEMBLE_SIZE = 5  # Number of models to be trained for the ensemble
EPOCHS = 1  # Give it a bit number (>1000)
BATCH_SIZE = 10
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
SHUFFLE_BUFFER = 500



###
### Inference parameters
###
OUTPUT_MASK_RGB_COLOR = [0, 255, 0]  # RGB value as a list
THRESHOLD_VALUE = 0.5
